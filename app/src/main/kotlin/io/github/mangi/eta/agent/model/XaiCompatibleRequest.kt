package io.github.mangi.eta.agent.model

import io.github.mangi.eta.data.model.ProviderAuthModes
import io.github.mangi.eta.data.model.ProviderSourceTypes
import io.github.mangi.eta.data.provider.ProviderSourceRegistry
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import org.json.JSONArray
import org.json.JSONObject

/**
 * xAI 编译工具参数时，根 schema 必须是 object，或每个分支都直接是 object 的 anyOf/oneOf。
 * 只有 required 之类约束、没有 type=object 的分支会被当成非对象分支，整次对话返回 HTTP 400。
 * pattern / format 在 Responses 路径上也会让同一请求失败。
 * grok-4.5 及更新模型不能关闭推理，reasoning_effort=none 以及 presence/frequency/stop 同样返回 400。
 * 只改写出站副本，工具目录和 OpenAI 请求保持原样。
 */
internal object XaiCompatibleRequest {
    private val strippedKeywords = setOf(
        "pattern",
        "format",
        "minContains",
        "maxContains",
        "uniqueItems",
    )
    private val unionKeywords = listOf("anyOf", "oneOf")
    private val schemaArrayKeywords = listOf("anyOf", "oneOf", "allOf", "prefixItems")
    private val schemaMapKeywords = listOf(
        "properties",
        "\$defs",
        "definitions",
        "patternProperties",
        "dependentSchemas",
    )
    private val schemaObjectKeywords = listOf(
        "additionalProperties",
        "unevaluatedProperties",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "contains",
    )
    private val parameterKeys = listOf("parameters", "parametersJsonSchema")
    private val acceptedEfforts = setOf("none", "low", "medium", "high", "xhigh")
    private val reasoningPenaltyFields = setOf("presence_penalty", "frequency_penalty", "stop")

    fun prepare(request: JSONObject, config: AgentModelClient.ModelConfig) {
        if (!appliesTo(config)) return
        stripUnsupportedFields(request, config.model)
        val tools = request.optJSONArray("tools") ?: return
        request.put("tools", sanitizeTools(tools))
    }

    private fun appliesTo(config: AgentModelClient.ModelConfig): Boolean {
        if (config.authMode == ProviderAuthModes.XAI_GROK) return true
        val source = ProviderSourceRegistry.resolve(
            providerId = config.providerId,
            sourceType = config.providerSourceType,
            baseUrl = config.baseUrl,
            providerType = config.providerType,
        )
        if (source == ProviderSourceTypes.XAI) return true
        val host = config.baseUrl.trim().toHttpUrlOrNull()?.host
        return host == "api.x.ai" || host == "cli-chat-proxy.grok.com"
    }

    private fun stripUnsupportedFields(request: JSONObject, model: String) {
        val id = modelId(model)
        if (isGrok4(id)) {
            reasoningPenaltyFields.forEach(request::remove)
        } else if (id == "grok-3" || id.startsWith("grok-3-") || id.startsWith("grok-3.")) {
            request.remove("presence_penalty")
        }
        if (!isGrok4(id)) return
        if (request.has("reasoning_effort") && !acceptedEffort(request.optString("reasoning_effort"), id)) {
            request.remove("reasoning_effort")
        }
        val reasoning = request.optJSONObject("reasoning") ?: return
        if (reasoning.has("effort") && !acceptedEffort(reasoning.optString("effort"), id)) {
            reasoning.remove("effort")
        }
        if (reasoning.length() == 0) request.remove("reasoning")
    }

    private fun acceptedEffort(effort: String, modelId: String): Boolean {
        if (effort !in acceptedEfforts) return false
        if (effort == "none" && !allowsDisabledReasoning(modelId)) return false
        return true
    }

    private fun modelId(model: String): String =
        model.trim().lowercase().substringAfter('/').substringBefore(':')

    private fun isGrok4(id: String): Boolean =
        id == "grok-4" || id.startsWith("grok-4-") || id.startsWith("grok-4.")

    /** grok-4.3 的 Chat Completions 接受 reasoning_effort=none；4.5 及更新模型不能关闭推理。 */
    private fun allowsDisabledReasoning(id: String): Boolean =
        id == "grok-4.3" || id.startsWith("grok-4.3-") || id.startsWith("grok-4.3.")

    private fun sanitizeTools(tools: JSONArray): JSONArray {
        val copy = JSONArray(tools.toString())
        for (index in 0 until copy.length()) {
            val tool = copy.optJSONObject(index) ?: continue
            parameterHolders(tool).forEach(::sanitizeParameterHolder)
        }
        return copy
    }

    private fun parameterHolders(tool: JSONObject): List<JSONObject> {
        val holders = mutableListOf<JSONObject>()
        tool.optJSONObject("function")?.let(holders::add)
        if (parameterKeys.any(tool::has)) holders.add(tool)
        return holders
    }

    private fun sanitizeParameterHolder(holder: JSONObject) {
        for (key in parameterKeys) {
            if (!holder.has(key) || holder.isNull(key)) continue
            val schema = holder.optJSONObject(key)
            if (schema == null) {
                holder.put(key, permissiveObject())
            } else {
                sanitizeSchema(schema, parameterRoot = true)
            }
        }
    }

    private fun sanitizeSchema(schema: JSONObject, parameterRoot: Boolean) {
        strippedKeywords.forEach(schema::remove)
        for (key in schemaMapKeywords) {
            schema.optJSONObject(key)?.let { map ->
                sanitizeSchemaMap(if (key == "properties") schema else null, map)
            }
        }
        rewriteItems(schema)
        schema.optJSONObject("items")?.let { sanitizeSchema(it, parameterRoot = false) }
        for (key in schemaObjectKeywords) {
            schema.optJSONObject(key)?.let { sanitizeSchema(it, parameterRoot = false) }
        }
        for (key in schemaArrayKeywords) {
            val array = schema.optJSONArray(key) ?: continue
            if (array.length() == 0 && key != "prefixItems") {
                schema.remove(key)
                continue
            }
            for (index in 0 until array.length()) {
                array.optJSONObject(index)?.let { sanitizeSchema(it, parameterRoot = false) }
            }
        }
        val enum = schema.optJSONArray("enum")
        if (enum != null && enum.length() == 0) schema.remove("enum")
        if (parameterRoot) normalizeParameterRoot(schema)
    }

    private fun sanitizeSchemaMap(parent: JSONObject?, map: JSONObject) {
        val removed = mutableListOf<String>()
        for (name in map.keys().asSequence().toList()) {
            val child = map.opt(name)
            if (child is JSONObject) {
                sanitizeSchema(child, parameterRoot = false)
            } else if (child == true) {
                map.put(name, JSONObject())
            } else {
                map.remove(name)
                removed += name
            }
        }
        if (removed.isEmpty() || parent?.optJSONArray("required") == null) return
        val required = parent.getJSONArray("required")
        val kept = JSONArray()
        for (index in 0 until required.length()) {
            val name = required.optString(index)
            if (name !in removed) kept.put(name)
        }
        if (kept.length() == 0) parent.remove("required") else parent.put("required", kept)
    }

    private fun rewriteItems(schema: JSONObject) {
        val items = schema.opt("items")
        if (items is JSONArray) {
            schema.remove("items")
            if (!schema.has("prefixItems")) schema.put("prefixItems", items)
        } else if (items is Boolean) {
            if (items) schema.put("items", JSONObject()) else schema.remove("items")
        }
    }

    private fun normalizeParameterRoot(schema: JSONObject) {
        val unions = unionKeywords.mapNotNull { key ->
            schema.optJSONArray(key)?.let { key to it }
        }
        if (unions.isNotEmpty()) {
            val promotable = unions.all { (_, branches) ->
                branches.length() > 0 &&
                    (0 until branches.length()).all { index -> promotableObjectBranch(branches.opt(index)) }
            }
            if (promotable) {
                unions.forEach { (_, branches) ->
                    for (index in 0 until branches.length()) markObject(branches.getJSONObject(index))
                }
                val type = schema.opt("type")
                if (schema.has("type") && !schema.isNull("type") && type != null && !isObjectType(type)) {
                    schema.put("type", "object")
                }
                return
            }
            unions.forEach { (key, _) -> schema.remove(key) }
        }
        if (isObjectSchema(schema) || schema.has("allOf") || schema.has("\$ref")) {
            if (!schema.has("type") || schema.isNull("type")) schema.put("type", "object")
            return
        }
        val description = schema.optString("description").takeIf { schema.has("description") && !schema.isNull("description") }
        schema.keys().asSequence().toList().forEach(schema::remove)
        schema.put("type", "object")
        schema.put("additionalProperties", true)
        if (!description.isNullOrEmpty()) schema.put("description", description)
    }

    private fun promotableObjectBranch(value: Any?): Boolean {
        val branch = value as? JSONObject ?: return false
        if (unionKeywords.any(branch::has) || branch.has("allOf") || branch.has("\$ref")) return false
        if (!branch.has("type") || branch.isNull("type")) return true
        return isObjectType(branch.opt("type"))
    }

    private fun markObject(branch: JSONObject) {
        if (branch.opt("type") == "object") return
        branch.put("type", "object")
    }

    private fun isObjectSchema(schema: JSONObject): Boolean {
        if (schema.has("type") && !schema.isNull("type")) return isObjectType(schema.opt("type"))
        return schema.has("properties") || schema.has("required") || schema.has("additionalProperties")
    }

    private fun isObjectType(type: Any?): Boolean = when (type) {
        "object" -> true
        is JSONArray -> type.length() > 0 && (0 until type.length()).all { index -> type.optString(index) == "object" }
        else -> false
    }

    private fun permissiveObject(): JSONObject =
        JSONObject().put("type", "object").put("additionalProperties", true)
}
