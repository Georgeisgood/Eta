package io.github.mangi.eta.agent.model

import io.github.mangi.eta.data.model.ModelReasoningCapabilities
import io.github.mangi.eta.data.model.OpenAiEndpointMode
import io.github.mangi.eta.data.model.ProviderAuthModes
import io.github.mangi.eta.data.model.ProviderSourceTypes
import io.github.mangi.eta.data.model.ReasoningEffort
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class XaiCompatibleRequestTest {
    @Test
    fun chatCompletionsPromoteConstraintOnlyRootUnionsAndKeepOpenAiUntouched() {
        val tools = JSONArray().put(fetchUrlTool())
        val original = tools.toString()
        val xai = OpenAiChatCompletionsProvider.buildRequestJson(
            config(ProviderSourceTypes.XAI, "https://api.x.ai/v1", "grok-4.7"),
            JSONArray(),
            tools,
        )
        val openAi = OpenAiChatCompletionsProvider.buildRequestJson(
            config(ProviderSourceTypes.OPENAI, "https://api.openai.com/v1", "gpt-5.5"),
            JSONArray(),
            tools,
        )

        assertEquals(original, tools.toString())
        val promoted = parameters(xai.getJSONArray("tools"), "fetch_url").getJSONArray("oneOf")
        assertEquals("object", promoted.getJSONObject(0).getString("type"))
        assertEquals(JSONArray().put("url").toString(), promoted.getJSONObject(0).getJSONArray("required").toString())
        assertEquals("object", promoted.getJSONObject(1).getString("type"))
        assertFalse(parameters(openAi.getJSONArray("tools"), "fetch_url").getJSONArray("oneOf").getJSONObject(0).has("type"))
    }

    @Test
    fun responsesToolsAndChatToolsStripPatternFormatAndCollapseNonObjectRoots() {
        val parameters = JSONObject()
            .put("oneOf", JSONArray()
                .put(JSONObject().put("type", "object").put("properties", JSONObject().put("mode", JSONObject().put("type", "string"))))
                .put(JSONObject().put("oneOf", JSONArray().put(JSONObject().put("type", "null")))))
            .put("description", "variants")
        val tools = JSONArray().put(
            JSONObject()
                .put("type", "function")
                .put("function", JSONObject()
                    .put("name", "automation_update")
                    .put("parameters", parameters)),
        )
        val chat = OpenAiChatCompletionsProvider.buildRequestJson(
            config(ProviderSourceTypes.XAI, "https://api.x.ai/v1", "grok-4.6"),
            JSONArray(),
            tools,
        )
        val sent = parameters(chat.getJSONArray("tools"), "automation_update")
        assertFalse(sent.has("oneOf"))
        assertEquals("object", sent.getString("type"))
        assertEquals("variants", sent.getString("description"))
        assertTrue(sent.getBoolean("additionalProperties"))

        val responsesTools = JSONArray().put(
            JSONObject()
                .put("type", "function")
                .put("function", JSONObject()
                    .put("name", "lookup")
                    .put("description", "lookup")
                    .put("parameters", JSONObject()
                        .put("type", "object")
                        .put("properties", JSONObject()
                            .put("query", JSONObject()
                                .put("type", "string")
                                .put("pattern", "\\bfoo")
                                .put("format", "date-time")
                                .put("anyOf", JSONArray()
                                    .put(JSONObject().put("type", "string").put("format", "email"))
                                    .put(JSONObject().put("type", "null"))))
                            .put("extra", true)
                            .put("blocked", false))
                        .put("\$defs", JSONObject()
                            .put("id", JSONObject().put("type", "string").put("format", "uuid").put("pattern", "^[a-z]+$")))
                        .put("items", JSONArray().put(JSONObject().put("type", "string").put("pattern", "a+"))))),
        )
        val responses = ResponsesRequestBuilder.build(
            config(ProviderSourceTypes.XAI, "https://api.x.ai/v1", "grok-4.7")
                .copy(openAiEndpointMode = OpenAiEndpointMode.RESPONSES),
            JSONArray(),
            responsesTools,
        )
        val schema = parameters(responses.getJSONArray("tools"), "lookup")
        assertFalse(containsKeyword(schema, "pattern"))
        assertFalse(containsKeyword(schema, "format"))
        assertFalse(schema.has("items"))
        assertEquals("string", schema.getJSONArray("prefixItems").getJSONObject(0).getString("type"))
        val query = schema.getJSONObject("properties").getJSONObject("query")
        assertEquals(2, query.getJSONArray("anyOf").length())
        assertEquals("null", query.getJSONArray("anyOf").getJSONObject(1).getString("type"))
        assertTrue(schema.getJSONObject("properties").has("extra"))
        assertFalse(schema.getJSONObject("properties").has("blocked"))
        assertFalse(schema.getJSONObject("\$defs").getJSONObject("id").has("format"))
    }

    @Test
    fun defaultAgentCatalogIsSafeForGrokAndUnchangedForGpt() {
        val tools = AgentToolCatalog.build(
            terminalTools = true,
            browserTools = true,
            deviceDirectTools = true,
            deviceSensitiveReadTools = true,
            deviceSensitiveActionTools = true,
            skillGitHubDiscovery = true,
            skillGitHubInstall = true,
        )
        val original = tools.toString()
        assertTrue(
            "xAI accepts at most 128 tools; default catalog has ${tools.length()}",
            tools.length() <= 128,
        )
        val xai = OpenAiChatCompletionsProvider.buildRequestJson(
            config(ProviderSourceTypes.XAI, "https://api.x.ai/v1", "grok-4.7", ReasoningEffort.OFF),
            JSONArray().put(JSONObject().put("role", "user").put("content", "hi")),
            tools,
        )
        val gpt = OpenAiChatCompletionsProvider.buildRequestJson(
            config(ProviderSourceTypes.OPENAI, "https://api.openai.com/v1", "gpt-5.5", ReasoningEffort.OFF),
            JSONArray(),
            tools,
        )

        assertEquals(original, tools.toString())
        assertFalse(xai.has("reasoning_effort"))
        assertFalse(xai.has("presence_penalty"))
        val sent = xai.getJSONArray("tools")
        assertEquals(tools.length(), sent.length())
        for (index in 0 until sent.length()) {
            val schema = sent.getJSONObject(index).getJSONObject("function").getJSONObject("parameters")
            assertParameterRoot(schema, sent.getJSONObject(index).getJSONObject("function").getString("name"))
            assertFalse(containsKeyword(schema, "pattern"))
            assertFalse(containsKeyword(schema, "format"))
            assertFalse(containsKeyword(schema, "uniqueItems"))
        }
        val fetch = parameters(sent, "fetch_url")
        assertEquals("object", fetch.getJSONArray("oneOf").getJSONObject(0).getString("type"))
        assertEquals("object", fetch.getJSONArray("oneOf").getJSONObject(1).getString("type"))
        assertEquals(1, parameters(sent, "web_search").getJSONObject("properties").getJSONObject("query").getInt("minLength"))
        assertFalse(
            parameters(gpt.getJSONArray("tools"), "fetch_url").getJSONArray("oneOf").getJSONObject(0).has("type"),
        )
        assertEquals("none", gpt.getString("reasoning_effort"))
    }

    @Test
    fun grok46DropsDisabledReasoningAndSamplingPenaltiesWhileGrok43KeepsNone() {
        val penalties = """{"presence_penalty":0.6,"frequency_penalty":0.2,"stop":["END"],"temperature":0.3}"""
        val disabled = OpenAiChatCompletionsProvider.buildRequestJson(
            config(ProviderSourceTypes.XAI, "https://api.x.ai/v1", "grok-4.6", ReasoningEffort.OFF)
                .copy(extraBodyJson = penalties),
            JSONArray(),
            JSONArray(),
        )
        assertFalse(disabled.has("reasoning_effort"))
        assertFalse(disabled.has("presence_penalty"))
        assertFalse(disabled.has("frequency_penalty"))
        assertFalse(disabled.has("stop"))
        assertEquals(0.3, disabled.getDouble("temperature"), 0.0)
        assertEquals(0, disabled.getJSONArray("tools").length())

        val kept = OpenAiChatCompletionsProvider.buildRequestJson(
            config(ProviderSourceTypes.XAI, "https://api.x.ai/v1", "grok-4.3", ReasoningEffort.OFF)
                .copy(extraBodyJson = penalties),
            JSONArray(),
            JSONArray(),
        )
        assertEquals("none", kept.getString("reasoning_effort"))
        assertFalse(kept.has("presence_penalty"))
        assertEquals(0.3, kept.getDouble("temperature"), 0.0)

        val responses = ResponsesRequestBuilder.build(
            config(ProviderSourceTypes.XAI, "https://api.x.ai/v1", "grok-4.7", ReasoningEffort.OFF)
                .copy(openAiEndpointMode = OpenAiEndpointMode.RESPONSES, extraBodyJson = penalties),
            JSONArray(),
            JSONArray(),
        )
        assertFalse(responses.has("reasoning"))
        assertFalse(responses.has("presence_penalty"))
        assertFalse(responses.has("stop"))

        val grok43Responses = ResponsesRequestBuilder.build(
            config(ProviderSourceTypes.XAI, "https://api.x.ai/v1", "grok-4.3", ReasoningEffort.OFF)
                .copy(openAiEndpointMode = OpenAiEndpointMode.RESPONSES),
            JSONArray(),
            JSONArray(),
        )
        assertEquals("none", grok43Responses.getJSONObject("reasoning").getString("effort"))
    }

    @Test
    fun nonXaiHostsAndOpenRouterKeepSchemasEvenWhenTheModelNameIsGrok() {
        val tools = JSONArray().put(fetchUrlTool())
        val request = JSONObject().put("tools", tools).put("presence_penalty", 1).put("reasoning_effort", "none")
        XaiCompatibleRequest.prepare(
            request,
            config(ProviderSourceTypes.OPENROUTER, "https://openrouter.ai/api/v1", "x-ai/grok-4.7"),
        )
        assertTrue(request.get("tools") === tools)
        assertEquals(1, request.getInt("presence_penalty"))
        assertFalse(parameters(tools, "fetch_url").getJSONArray("oneOf").getJSONObject(0).has("type"))

        val proxy = JSONObject().put("tools", JSONArray(tools.toString())).put("reasoning_effort", "none")
        XaiCompatibleRequest.prepare(
            proxy,
            config(ProviderSourceTypes.CUSTOM, "https://cli-chat-proxy.grok.com/v1", "grok-4.6"),
        )
        assertEquals(
            "object",
            parameters(proxy.getJSONArray("tools"), "fetch_url").getJSONArray("oneOf").getJSONObject(0).getString("type"),
        )
        assertFalse(proxy.has("reasoning_effort"))
    }

    @Test
    fun xaiOAuthStillSanitizesWhenSourceTypeIsCustom() {
        val tools = JSONArray().put(fetchUrlTool())
        val request = OpenAiChatCompletionsProvider.buildRequestJson(
            config(ProviderSourceTypes.CUSTOM, "https://example.invalid/v1", "grok-4.7", ReasoningEffort.HIGH)
                .copy(authMode = ProviderAuthModes.XAI_GROK),
            JSONArray(),
            tools,
        )
        assertEquals(
            "object",
            parameters(request.getJSONArray("tools"), "fetch_url").getJSONArray("oneOf").getJSONObject(0).getString("type"),
        )
        assertEquals("high", request.getString("reasoning_effort"))
    }

    private fun fetchUrlTool(): JSONObject =
        JSONObject()
            .put("type", "function")
            .put("function", JSONObject()
                .put("name", "fetch_url")
                .put("description", "read")
                .put("parameters", JSONObject()
                    .put("type", "object")
                    .put("properties", JSONObject()
                        .put("url", JSONObject().put("type", "string").put("minLength", 1))
                        .put("document_id", JSONObject().put("type", "string")))
                    .put("oneOf", JSONArray()
                        .put(JSONObject().put("required", JSONArray().put("url")))
                        .put(JSONObject().put("required", JSONArray().put("document_id"))))
                    .put("additionalProperties", false)))

    private fun config(
        source: String,
        baseUrl: String,
        model: String,
        effort: ReasoningEffort = ReasoningEffort.OFF,
    ) = AgentModelClient.ModelConfig(
        providerId = if (source == ProviderSourceTypes.XAI) "builtin-xai" else "",
        providerSourceType = source,
        baseUrl = baseUrl,
        apiKey = "test-key",
        model = model,
        systemPrompt = "",
        reasoningEffort = effort,
        reasoningCapabilities = ModelReasoningCapabilities(
            supportedEfforts = listOf(ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH, ReasoningEffort.XHIGH),
            canDisable = true,
        ),
    )

    private fun parameters(tools: JSONArray, name: String): JSONObject {
        for (index in 0 until tools.length()) {
            val tool = tools.getJSONObject(index)
            val function = tool.optJSONObject("function")
            val toolName = function?.optString("name")?.takeIf { it.isNotBlank() } ?: tool.optString("name")
            if (toolName != name) continue
            return function?.optJSONObject("parameters") ?: tool.getJSONObject("parameters")
        }
        error("missing tool $name")
    }

    private fun assertParameterRoot(schema: JSONObject, name: String) {
        val unions = unionKeywords.mapNotNull { key -> schema.optJSONArray(key) }
        if (unions.isEmpty()) {
            assertEquals("$name root must be an object", "object", schema.optString("type"))
            return
        }
        for (union in unions) {
            assertTrue("$name union must be non-empty", union.length() > 0)
            for (index in 0 until union.length()) {
                assertEquals(
                    "$name branch $index must be an object",
                    "object",
                    union.getJSONObject(index).optString("type"),
                )
            }
        }
    }

    private fun containsKeyword(schema: JSONObject, keyword: String): Boolean {
        if (schema.has(keyword)) return true
        val maps = listOf("properties", "\$defs", "definitions", "patternProperties", "dependentSchemas")
        for (key in maps) {
            val map = schema.optJSONObject(key) ?: continue
            for (name in map.keys()) {
                val child = map.optJSONObject(name) ?: continue
                if (containsKeyword(child, keyword)) return true
            }
        }
        for (key in listOf("items", "additionalProperties", "not", "if", "then", "else", "contains")) {
            val child = schema.optJSONObject(key) ?: continue
            if (containsKeyword(child, keyword)) return true
        }
        for (key in listOf("anyOf", "oneOf", "allOf", "prefixItems")) {
            val array = schema.optJSONArray(key) ?: continue
            for (index in 0 until array.length()) {
                val child = array.optJSONObject(index) ?: continue
                if (containsKeyword(child, keyword)) return true
            }
        }
        return false
    }

    private val unionKeywords = listOf("anyOf", "oneOf")
}
