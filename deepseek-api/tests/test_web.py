import http.client
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import agent
import day_three
import errors
import web


class DeepSeekWebTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.answers = []
        self.answer = "Тестовый ответ"
        self.model_error = None
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.state_path = Path(self.temporary_directory.name) / "agents.json"
        self.environment = patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"})
        self.environment.start()
        self.server = web.ChatServer(("127.0.0.1", 0), self.ask_model, self.state_path)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.environment.stop()
        self.temporary_directory.cleanup()

    def ask_model(self, payload, **kwargs):
        self.calls.append({"payload": payload, "kwargs": kwargs})
        if self.model_error:
            raise self.model_error
        if self.answers:
            return self.answers.pop(0)
        return self.answer

    def request(self, method, path, payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port)
        body = None if payload is None else json.dumps(payload)
        request_headers = headers or {}
        if body is not None:
            request_headers = {"Content-Type": "application/json", **request_headers}
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        content = response.read().decode("utf-8")
        response_headers = dict(response.getheaders())
        connection.close()
        return response.status, content, response_headers

    def json_request(self, method, path, payload=None, headers=None):
        status, body, response_headers = self.request(method, path, payload, headers)
        return status, json.loads(body), response_headers

    def css_block(self, page, selector, start=0):
        opening = page.find(f"{selector} {{", start)
        self.assertNotEqual(opening, -1, f"Missing CSS block for {selector}.")
        closing = page.index("}", opening)
        return page[opening:closing + 1]

    def test_page_contains_chat_controls(self):
        status, body, headers = self.request("GET", "/")

        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        app_css = self.css_block(body, ".app")
        chat_css = self.css_block(body, ".chat")
        header_css = self.css_block(body, "header")
        thread_css = self.css_block(body, ".thread")
        composer_css = self.css_block(body, ".composer-area")
        mobile_start = body.index("@media (max-width:860px)")
        mobile_css = body[mobile_start:body.index("</style>", mobile_start)]
        mobile_app_css = self.css_block(mobile_css, ".app")
        mobile_chat_css = self.css_block(mobile_css, ".chat")

        self.assertIn("height:100vh", app_css)
        self.assertIn("overflow:hidden", app_css)
        self.assertIn("height:100vh", chat_css)
        self.assertIn("display:flex", chat_css)
        self.assertIn("flex-direction:column", chat_css)
        self.assertIn("min-height:0", chat_css)
        self.assertIn("flex:0 0 auto", header_css)
        self.assertIn("flex:0 0 auto", composer_css)
        self.assertIn("flex:1", thread_css)
        self.assertIn("min-height:0", thread_css)
        self.assertIn("overflow:auto", thread_css)
        self.assertIn("height:auto", mobile_app_css)
        self.assertIn("overflow:visible", mobile_app_css)
        self.assertIn("height:100vh", mobile_chat_css)
        self.assertIn("Создать агента", body)
        self.assertIn("Агенты", body)
        self.assertNotIn('id="agent-count-input"', body)
        self.assertNotIn('id="create-many-agents"', body)
        self.assertIn('id="agent-list"', body)
        self.assertIn('<script src="/static/agent-state.js"></script>', body)
        self.assertIn('setAttribute("aria-label","Удалить агента")', body)
        self.assertIn('.agent-row.active { background:#f3dfd4', body)
        self.assertIn('.agent-row.active .session { background:transparent', body)
        self.assertIn('row.className="agent-row"+(agent.id===state.activeId?" active":"")', body)
        self.assertIn('window.confirm("Удалить агента "+agent.name+"? Это действие нельзя отменить.")', body)
        self.assertIn('method:"DELETE"', body)
        self.assertIn('"/api/agents/"+agent.id', body)
        self.assertIn('AgentClientState.agentExists(state,pending.agentId)', body)
        self.assertIn('AgentClientState.agentExists(state,agentId)', body)
        self.assertIn("Метаданные", body)
        self.assertIn("message-input", body)
        self.assertNotIn("DEEPSEEK_API_KEY", body)

    def test_root_serves_the_static_index_page(self):
        static_page = Path(web.__file__).with_name("static") / "index.html"

        status, body, headers = self.request("GET", "/")

        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertEqual(body, static_page.read_text(encoding="utf-8"))

    def test_static_agent_state_script_is_served(self):
        static_script = Path(web.__file__).with_name("static") / "agent-state.js"

        status, body, headers = self.request("GET", "/static/agent-state.js")

        self.assertEqual(status, 200)
        self.assertIn("application/javascript", headers["Content-Type"])
        self.assertEqual(body, static_script.read_text(encoding="utf-8"))

    def test_page_contains_response_settings_and_metadata(self):
        status, body, _ = self.request("GET", "/")

        self.assertEqual(status, 200)
        self.assertIn("Настройки ответа", body)
        self.assertIn("system-prompt-input", body)
        self.assertIn("response-format-input", body)
        self.assertIn("max-tokens-input", body)
        self.assertIn("stop-input", body)
        self.assertIn("scheduleSettingsSave", body)
        self.assertNotIn('id="save-settings"', body)
        self.assertNotIn("format-instruction-input", body)
        self.assertNotIn("Инструкция свободного формата", body)
        self.assertIn("Метаданные", body)

    def test_page_exposes_four_context_modes_and_summary_metrics(self):
        status, body, _ = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn('/static/context-controls.js', body)
        status, controls, _ = self.request("GET", "/static/context-controls.js")
        self.assertEqual(status, 200)
        for mode in ('sliding_window', 'facts', 'branching', 'summary'):
            self.assertIn(f'value="{mode}"', controls)
        for label in ("Сжато сообщений", "Размер summary", "Токены: полная / сжатая история", "Стоимость summary"):
            self.assertIn(label, body)

    def test_page_contains_model_and_usage_metadata(self):
        status, body, _ = self.request("GET", "/")

        self.assertEqual(status, 200)
        self.assertIn("model-input", body)
        self.assertIn("Время ответа", body)
        self.assertIn("Токены", body)
        self.assertIn("Стоимость запроса", body)

    def test_page_contains_structured_token_metrics(self):
        status, body, _ = self.request("GET", "/")

        self.assertEqual(status, 200)
        for marker in (
            '"token-summary"', '"last-step-metrics"',
            'token-metrics-list', 'token-metric-row', 'id="context-attempt-card"',
            'id="token-history-scroll"', 'id="token-history-table"',
            'context-demo-row',
        ):
            self.assertIn(marker, body)
        for label in (
            "Метрики чата", "Сообщений", "Стоимость",
            "Токенов в сообщении", "Токенов на входе",
            "Токенов на выходе", "Токенов в ответе", "Вся история (накоплено)",
            "Контекст / лимит",
        ):
            self.assertIn(label, body)
        self.assertNotIn('"История до"', body)
        self.assertNotIn("token-metric-card", body)
        self.assertNotIn("context-limit-card", body)
        self.assertNotIn("≈", body)
        self.assertIn("Рост по ходам", body)
        self.assertIn("История по ходам", body)
        for header in ("Ход", "Вход", "Выход", "Всего", "Накопленные токены", "Стоимость"):
            self.assertIn(header, body)

    def test_page_handles_token_chart_edge_cases(self):
        status, body, _ = self.request("GET", "/")

        self.assertEqual(status, 200)
        self.assertIn('id="token-growth-chart"', body)
        self.assertIn("Данных пока нет", body)
        self.assertIn("metrics.calls.map(call=>call.promptTokens)", body)
        self.assertIn("Math.max(values.length-1,1)", body)

    def test_agent_chat_feedback_and_send_button_follow_the_selected_agent(self):
        status, body, _ = self.request("GET", "/")

        self.assertEqual(status, 200)
        self.assertIn("button.append(title,detail);", body)
        self.assertIn('button.addEventListener("click",async()=>', body)
        self.assertIn("/api/agents", body)
        self.assertNotIn("/api/agents/bulk", body)
        self.assertNotIn("/api/sessions", body)
        self.assertNotIn('id="status"', body)
        self.assertNotIn("function renderStatus()", body)
        self.assertIn("pendingAgentIds:new Set()", body)
        self.assertIn("drafts:{}", body)
        self.assertIn("function appendChatStatus", body)
        self.assertIn("state.pendingAgentIds.has(state.activeId)", body)
        self.assertIn("function addOptimisticUserMessage", body)
        self.assertIn("function renderComposer()", body)
        self.assertIn("const agentId=state.activeId;", body)
        self.assertIn("state.pendingAgentIds.add(agentId); contextControls.refresh(); addOptimisticUserMessage(agentId,text);", body)
        self.assertIn('"/api/agents/"+agentId+"/messages"', body)
        self.assertLess(body.index("addOptimisticUserMessage(agentId,text)"), body.index('"/api/agents/"+agentId+"/messages"'))
        self.assertIn('id="send" type="submit" disabled', body)

    def test_initial_agent_is_available(self):
        status, body, headers = self.json_request("GET", "/api/agents")

        self.assertEqual(status, 200)
        self.assertIn("application/json", headers["Content-Type"])
        self.assertEqual(
            body,
            {"agents": [{"id": "agent-1", "name": "Агент 1", "messages": [], "settings": {
                "model": agent.MODEL,
                "systemPrompt": agent.SYSTEM_PROMPT,
                "format": "text",
                "maxTokens": None,
                "stop": "",
                "trainingContextLimit": None,
                "sendOnOverflow": False,
                "contextCompressionEnabled": True,
                "contextStrategy": "sliding_window", "windowSize": 10,
            }, "metadata": None, "metrics": agent.default_metrics(), "context": agent.Agent("agent-1", "Агент 1", agent.default_settings(), lambda p: "").snapshot()["context"]}]},
        )

    def test_server_restart_restores_agent_context(self):
        settings = {
            "model": "deepseek-v4-pro",
            "systemPrompt": "Отвечай по имени.",
            "format": "text",
            "maxTokens": None,
            "stop": "",
        }
        self.json_request("PUT", "/api/agents/agent-1/settings", settings)
        self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Меня зовут Маша"})

        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.server = web.ChatServer(("127.0.0.1", 0), self.ask_model, self.state_path)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        status, _, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Как меня зовут?"})

        self.assertEqual(status, 200)
        self.assertEqual(self.calls[-1]["payload"], {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "system", "content": "Отвечай по имени."},
                {"role": "user", "content": "Меня зовут Маша"},
                {"role": "assistant", "content": "Тестовый ответ"},
                {"role": "user", "content": "Как меня зовут?"},
            ],
            "temperature": 1,
        })

    def test_storage_error_returns_safe_500(self):
        with patch.object(
            agent.AgentRegistry,
            "_save",
            side_effect=agent.PersistenceError("Диск недоступен"),
        ):
            status, body, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Привет"})

        self.assertEqual(status, 500)
        self.assertEqual(body, {"error": "Не удалось сохранить состояние агента."})

    def test_create_agent_assigns_next_id(self):
        status, body, _ = self.json_request("POST", "/api/agents")

        self.assertEqual(status, 201)
        self.assertEqual(body["agent"], {"id": "agent-2", "name": "Агент 2", "messages": [], "settings": {
            "model": agent.MODEL,
            "systemPrompt": agent.SYSTEM_PROMPT,
            "format": "text",
            "maxTokens": None,
            "stop": "",
            "trainingContextLimit": None,
            "sendOnOverflow": False,
            "contextCompressionEnabled": True,
                "contextStrategy": "sliding_window", "windowSize": 10,
        }, "metadata": None, "metrics": agent.default_metrics(), "context": agent.Agent("agent-1", "Агент 1", agent.default_settings(), lambda p: "").snapshot()["context"]})

    def test_delete_agent_removes_its_saved_history_and_returns_id(self):
        _, created, _ = self.json_request("POST", "/api/agents")
        agent_id = created["agent"]["id"]
        self.json_request("POST", f"/api/agents/{agent_id}/messages", {"text": "Удаляемая история"})

        status, body, _ = self.json_request("DELETE", f"/api/agents/{agent_id}")

        self.assertEqual(status, 200)
        self.assertEqual(body, {"deletedId": agent_id})
        _, agents, _ = self.json_request("GET", "/api/agents")
        self.assertEqual([snapshot["id"] for snapshot in agents["agents"]], ["agent-1"])
        saved_state = self.state_path.read_text(encoding="utf-8")
        self.assertNotIn(agent_id, saved_state)
        self.assertNotIn("Удаляемая история", saved_state)

    def test_delete_missing_agent_returns_404(self):
        status, body, _ = self.json_request("DELETE", "/api/agents/missing")

        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "Агент не найден."})

    def test_delete_storage_error_returns_safe_500(self):
        with patch.object(
            agent.AgentRegistry,
            "delete",
            side_effect=agent.PersistenceError("Диск недоступен"),
        ):
            status, body, _ = self.json_request("DELETE", "/api/agents/agent-1")

        self.assertEqual(status, 500)
        self.assertEqual(body, {"error": "Не удалось сохранить состояние агента."})

    def test_delete_rejects_foreign_origin_and_accepts_matching_origin(self):
        status, body, _ = self.json_request(
            "DELETE", "/api/agents/agent-1", headers={"Origin": "http://evil.example"},
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "Запрос с другого источника запрещён."})

        status, body, _ = self.json_request(
            "DELETE", "/api/agents/agent-1", headers={"Origin": f"http://127.0.0.1:{self.port}"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body, {"deletedId": "agent-1"})

    def test_message_returns_404_when_agent_is_deleted_after_lookup(self):
        self._assert_request_returns_404_when_agent_is_deleted_after_lookup(
            "POST", "/api/agents/agent-1/messages", {"text": "Привет"},
        )

    def test_settings_returns_404_when_agent_is_deleted_after_lookup(self):
        self._assert_request_returns_404_when_agent_is_deleted_after_lookup(
            "PUT", "/api/agents/agent-1/settings", {
                "model": agent.MODEL,
                "systemPrompt": agent.SYSTEM_PROMPT,
                "format": "text",
                "maxTokens": None,
                "stop": "",
            },
        )

    def _assert_request_returns_404_when_agent_is_deleted_after_lookup(self, method, path, payload):
        original_get = self.server.registry.get
        lookup_finished = threading.Event()
        continue_request = threading.Event()
        response = {}

        def delayed_get(agent_id):
            result = original_get(agent_id)
            if agent_id == "agent-1" and not lookup_finished.is_set():
                lookup_finished.set()
                continue_request.wait(timeout=2)
            return result

        with patch.object(self.server.registry, "get", side_effect=delayed_get):
            request_thread = threading.Thread(
                target=lambda: response.update(result=self.json_request(method, path, payload)),
                daemon=True,
            )
            request_thread.start()
            self.assertTrue(lookup_finished.wait(timeout=2))
            status, _, _ = self.json_request("DELETE", "/api/agents/agent-1")
            self.assertEqual(status, 200)
            continue_request.set()
            request_thread.join(timeout=2)

        self.assertFalse(request_thread.is_alive())
        self.assertEqual(response["result"][:2], (404, {"error": "Агент не найден."}))

    def test_bulk_create_returns_requested_independent_agents(self):
        status, body, _ = self.json_request("POST", "/api/agents/bulk", {"count": 3})

        self.assertEqual(status, 201)
        self.assertEqual([agent["id"] for agent in body["agents"]], ["agent-2", "agent-3", "agent-4"])
        self.assertEqual([agent["name"] for agent in body["agents"]], ["Агент 2", "Агент 3", "Агент 4"])
        self.assertEqual(len(self.server.registry.agents()), 4)
        self.assertTrue(all(snapshot["settings"]["systemPrompt"] == agent.SYSTEM_PROMPT for snapshot in body["agents"]))
        self.assertEqual(self.calls, [])

    def test_bulk_create_rejects_invalid_count(self):
        for payload in ({}, {"count": 0}, {"count": 101}, {"count": True}, {"count": "3"}):
            with self.subTest(payload=payload):
                status, body, _ = self.json_request("POST", "/api/agents/bulk", payload)
                self.assertEqual(status, 400)
                self.assertIn("количество", body["error"].lower())

    def test_settings_are_stored_per_agent(self):
        settings = {
            "model": agent.MODEL,
            "systemPrompt": "Отвечай кратко.",
            "format": "json",
            "maxTokens": 300,
            "stop": "<END>",
        }

        status, body, _ = self.json_request("PUT", "/api/agents/agent-1/settings", settings)

        self.assertEqual(status, 200)
        self.assertEqual(body["agent"]["settings"], {**agent.default_settings(), **{
            **settings, "trainingContextLimit": None, "sendOnOverflow": False,
            "contextCompressionEnabled": True,
                "contextStrategy": "sliding_window", "windowSize": 10,
        }})

    def test_settings_accept_all_legacy_shapes_and_normalize_compression_flag(self):
        legacy = {
            "model": agent.MODEL,
            "systemPrompt": "Отвечай кратко.",
            "format": "text",
            "maxTokens": None,
            "stop": "",
        }
        day_eight = {**legacy, "trainingContextLimit": 1234, "sendOnOverflow": True}

        for payload, expected in (
            (legacy, {**legacy, "trainingContextLimit": None, "sendOnOverflow": False, "contextCompressionEnabled": True}),
            (day_eight, {**day_eight, "contextCompressionEnabled": True}),
            ({**legacy, "contextCompressionEnabled": False}, {**legacy, "trainingContextLimit": None, "sendOnOverflow": False, "contextCompressionEnabled": False}),
            ({**day_eight, "contextCompressionEnabled": False}, {**day_eight, "contextCompressionEnabled": False}),
        ):
            with self.subTest(fields=len(payload)):
                status, body, _ = self.json_request("PUT", "/api/agents/agent-1/settings", payload)
                self.assertEqual(status, 200)
                self.assertEqual(body["agent"]["settings"], {**agent.default_settings(), **expected, "contextStrategy": "branching" if expected.get("contextCompressionEnabled") is False else "sliding_window"})

    def test_invalid_compression_flag_does_not_replace_saved_settings(self):
        saved_settings = {
            "model": agent.MODEL,
            "systemPrompt": "Сохранённая настройка.",
            "format": "text",
            "maxTokens": None,
            "stop": "",
            "contextCompressionEnabled": False,
        }
        status, _, _ = self.json_request("PUT", "/api/agents/agent-1/settings", saved_settings)
        self.assertEqual(status, 200)

        invalid = {**saved_settings, "contextCompressionEnabled": "false"}
        status, body, _ = self.json_request("PUT", "/api/agents/agent-1/settings", invalid)

        self.assertEqual(status, 400)
        self.assertIn("логическим", body["error"])
        _, agents, _ = self.json_request("GET", "/api/agents")
        self.assertEqual(agents["agents"][0]["settings"], {**agent.default_settings(), **{
            **saved_settings, "contextStrategy": "branching",
            "trainingContextLimit": None,
            "sendOnOverflow": False,
        }})

    def test_agent_model_and_metadata_are_persisted(self):
        settings = {
            "systemPrompt": agent.SYSTEM_PROMPT,
            "format": "text",
            "maxTokens": None,
            "stop": "",
            "model": "deepseek-v4-pro",
        }
        self.answer = {
            "content": "Ответ сильной модели",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }
        self.json_request("PUT", "/api/agents/agent-1/settings", settings)

        status, body, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Привет"})
        _, agents, _ = self.json_request("GET", "/api/agents")

        self.assertEqual(status, 200)
        self.assertEqual(self.calls[-1]["payload"]["model"], "deepseek-v4-pro")
        self.assertEqual(body["metadata"]["usage"], {
            "promptTokens": 100, "completionTokens": 50, "totalTokens": 150, "source": "actual",
        })
        self.assertEqual(body["metadata"]["cost"], {"kind": "paid", "usd": 0.000165, "source": "actual"})
        self.assertGreaterEqual(body["metadata"]["responseTimeMs"], 0)
        self.assertEqual(agents["agents"][0]["metadata"], body["metadata"])

    def test_invalid_settings_do_not_replace_saved_values(self):
        valid_settings = {
            "model": agent.MODEL,
            "systemPrompt": "Отвечай кратко.",
            "format": "text",
            "maxTokens": None,
            "stop": "",
        }
        self.json_request("PUT", "/api/agents/agent-1/settings", valid_settings)

        status, body, _ = self.json_request(
            "PUT", "/api/agents/agent-1/settings",
            {"model": agent.MODEL, "systemPrompt": "   ", "format": "text", "maxTokens": 0, "stop": ""},
        )

        self.assertEqual(status, 400)
        self.assertIn("system prompt", body["error"].lower())
        _, agents, _ = self.json_request("GET", "/api/agents")
        self.assertEqual(agents["agents"][0]["settings"], {**agent.default_settings(), **{
            **valid_settings, "trainingContextLimit": None, "sendOnOverflow": False,
            "contextCompressionEnabled": True,
                "contextStrategy": "sliding_window", "windowSize": 10,
        }})

    def test_settings_for_unknown_agent_return_404(self):
        status, body, _ = self.json_request(
            "PUT", "/api/agents/missing/settings",
            {"model": agent.MODEL, "systemPrompt": "Отвечай кратко.", "format": "text", "maxTokens": None, "stop": ""},
        )

        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "Агент не найден."})

    def test_legacy_format_instruction_setting_is_rejected(self):
        valid_settings = {
            "model": agent.MODEL,
            "systemPrompt": "Базовая инструкция.",
            "format": "text",
            "maxTokens": None,
            "stop": "",
        }
        seed_status, _, _ = self.json_request("PUT", "/api/agents/agent-1/settings", valid_settings)
        self.assertEqual(seed_status, 200)

        status, body, _ = self.json_request(
            "PUT", "/api/agents/agent-1/settings",
            {
                "model": agent.MODEL,
                "systemPrompt": "Базовая инструкция.",
                "format": "text",
                "formatInstruction": "Кратко.",
                "maxTokens": None,
                "stop": "",
            },
        )

        self.assertEqual(status, 400)
        self.assertEqual(body, {"error": "Настройки имеют неверный формат."})
        _, agents, _ = self.json_request("GET", "/api/agents")
        self.assertEqual(agents["agents"][0]["settings"], {**agent.default_settings(), **{
            **valid_settings, "trainingContextLimit": None, "sendOnOverflow": False,
            "contextCompressionEnabled": True,
                "contextStrategy": "sliding_window", "windowSize": 10,
        }})

    def test_message_returns_answer_and_exact_metadata(self):
        status, body, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "  Привет  "})

        expected_messages = [
            {"role": "system", "content": agent.SYSTEM_PROMPT},
            {"role": "user", "content": "Привет"},
        ]
        self.assertEqual(status, 200)
        self.assertEqual(body["agent"]["id"], "agent-1")
        self.assertEqual(body["agent"]["name"], "Агент 1")
        self.assertEqual(body["agent"]["messages"], [
            {"role": "user", "content": "Привет"}, {"role": "assistant", "content": "Тестовый ответ"},
        ])
        self.assertEqual(body["agent"]["settings"]["model"], agent.MODEL)
        self.assertEqual(body["agent"]["metadata"], body["metadata"])
        self.assertEqual(self.calls, [{"payload": {
            "model": agent.MODEL,
            "messages": expected_messages,
            "temperature": 1,
        }, "kwargs": {}}])
        self.assertEqual(body["metadata"]["userPrompt"], "Привет")
        self.assertEqual(body["metadata"]["systemPrompt"], agent.SYSTEM_PROMPT)
        self.assertEqual(body["metadata"]["payload"], {
            "model": agent.MODEL, "messages": expected_messages, "temperature": 1,
        })
        self.assertEqual(body["metadata"]["status"], {"kind": "success", "label": "200 OK"})
        self.assertEqual(body["metadata"]["usage"]["source"], "estimated")
        self.assertEqual(body["metadata"]["cost"]["source"], "estimated")

    def test_second_message_sends_complete_history(self):
        self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Первый"})
        self.answer = "Второй ответ"
        status, _, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Второй"})

        self.assertEqual(status, 200)
        self.assertEqual(self.calls[-1]["payload"]["messages"], [
            {"role": "system", "content": agent.SYSTEM_PROMPT},
            {"role": "user", "content": "Первый"},
            {"role": "assistant", "content": "Тестовый ответ"},
            {"role": "user", "content": "Второй"},
        ])

    def test_json_settings_are_sent_to_the_model(self):
        settings = {
            "model": agent.MODEL,
            "systemPrompt": "Верни данные.",
            "format": "json",
            "maxTokens": 300,
            "stop": "<END>",
        }
        self.json_request("PUT", "/api/agents/agent-1/settings", settings)

        status, body, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Привет"})

        self.assertEqual(status, 200)
        self.assertEqual(self.calls[-1]["kwargs"], {
            "response_format": {"type": "json_object"},
            "max_tokens": 300,
            "stop": "<END>",
        })
        self.assertEqual(body["metadata"]["payload"]["response_format"], {"type": "json_object"})
        self.assertEqual(body["metadata"]["payload"]["max_tokens"], 300)
        self.assertEqual(body["metadata"]["payload"]["stop"], "<END>")
        self.assertEqual(
            body["metadata"]["systemPrompt"],
            "Верни данные.\n\nВерни только валидный JSON без Markdown-разметки.",
        )
        self.assertEqual(
            self.calls[-1]["payload"]["messages"][0]["content"],
            "Верни данные.\n\nВерни только валидный JSON без Markdown-разметки.",
        )

    def test_text_format_forwards_system_prompt_unchanged(self):
        settings = {
            "model": agent.MODEL,
            "systemPrompt": "Базовая инструкция.",
            "format": "text",
            "maxTokens": None,
            "stop": "",
        }
        self.json_request("PUT", "/api/agents/agent-1/settings", settings)

        status, body, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Привет"})

        self.assertEqual(status, 200)
        self.assertEqual(
            self.calls[-1]["payload"]["messages"][0]["content"],
            "Базовая инструкция.",
        )
        self.assertEqual(body["metadata"]["systemPrompt"], "Базовая инструкция.")

    def test_text_settings_omit_optional_model_parameters(self):
        status, body, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Привет"})

        self.assertEqual(status, 200)
        self.assertEqual(self.calls[-1]["kwargs"], {})
        self.assertNotIn("response_format", body["metadata"]["payload"])
        self.assertNotIn("max_tokens", body["metadata"]["payload"])
        self.assertNotIn("stop", body["metadata"]["payload"])

    def test_invalid_message_bodies_are_rejected(self):
        cases = [
            ({"text": "   "}, "Пустое сообщение."),
            ({}, "Поле text должно быть непустой строкой."),
            ({"text": 12}, "Поле text должно быть непустой строкой."),
        ]
        for payload, expected_error in cases:
            with self.subTest(payload=payload):
                status, body, _ = self.json_request("POST", "/api/agents/agent-1/messages", payload)
                self.assertEqual(status, 400)
                self.assertEqual(body, {"error": expected_error})

        status, body, _ = self.request("POST", "/api/agents/agent-1/messages", headers={"Content-Type": "application/json"})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body), {"error": "Некорректный JSON."})

    def test_unknown_agent_returns_json_404(self):
        status, body, _ = self.json_request("POST", "/api/agents/unknown/messages", {"text": "Привет"})

        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "Агент не найден."})

    def test_provider_error_preserves_agent_and_metadata(self):
        self.model_error = RuntimeError("сеть недоступна")
        status, body, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Привет"})

        self.assertEqual(status, 502)
        self.assertEqual(body["error"], "Не удалось получить ответ DeepSeek.")
        self.assertEqual(body["agent"]["messages"], [{"role": "user", "content": "Привет"}])
        self.assertEqual(body["metadata"]["status"], {"kind": "error", "label": "Ошибка API"})
        self.assertNotIn("сеть недоступна", json.dumps(body, ensure_ascii=False))

    def test_empty_provider_answer_returns_502(self):
        self.answer = ""
        status, body, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Привет"})

        self.assertEqual(status, 502)
        self.assertEqual(body["error"], "Не удалось получить ответ DeepSeek.")
        self.assertEqual(body["agent"]["messages"], [{"role": "user", "content": "Привет"}])

    def test_legacy_settings_preserve_full_history_and_map_compression_mode(self):
        self.server.registry.update_settings("agent-1", {**agent.default_settings(), "contextStrategy": "branching"})
        for n in range(8):
            self.server.registry.respond("agent-1", str(n))
        legacy = {key: value for key, value in agent.default_settings().items() if key not in {"contextStrategy", "windowSize"}}
        for enabled, mode in ((True, "summary"), (False, "branching")):
            status, body, _ = self.json_request("PUT", "/api/agents/agent-1/settings", {**legacy, "contextCompressionEnabled": enabled})
            self.assertEqual(status, 200)
            self.assertEqual(body["agent"]["settings"]["contextStrategy"], mode)
            self.assertEqual(len(body["agent"]["messages"]), 16)

    def test_context_api_creates_two_branches_and_switches_without_model_calls(self):
        settings = {**agent.default_settings(), "contextStrategy": "branching"}
        status, _, _ = self.json_request("PUT", "/api/agents/agent-1/settings", settings)
        self.assertEqual(status, 200)
        self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Начало"})
        status, cp, _ = self.json_request("POST", "/api/agents/agent-1/checkpoints", {"name": "Общая точка"})
        self.assertEqual(status, 200)
        ids = []
        for name in ("А", "Б"):
            status, branch, _ = self.json_request("POST", "/api/agents/agent-1/branches", {"name": name, "checkpointId": cp["checkpointId"]})
            self.assertEqual(status, 200)
            ids.append(branch["branchId"])
        self.json_request("POST", "/api/agents/agent-1/switch-branch", {"branchId": ids[0]})
        self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Только А"})
        status, branch, _ = self.json_request("POST", "/api/agents/agent-1/switch-branch", {"branchId": ids[1]})
        self.assertEqual(status, 200)
        self.assertEqual(len(branch["agent"]["messages"]), 2)
        self.assertEqual(len(self.calls), 2)

    def test_context_routes_validate_mode_fields_and_origin(self):
        for path, data, expected in (
            ("agent-1/checkpoints", {"name": "Точка"}, 400),
            ("missing/checkpoints", {"name": "Точка"}, 404),
            ("agent-1/branches", {"unknown": "x"}, 400),
        ):
            status, _, _ = self.json_request("POST", "/api/agents/" + path, data)
            self.assertEqual(status, expected)
        status, _, _ = self.json_request("POST", "/api/agents/agent-1/checkpoints", {"name": "Точка"}, headers={"Origin": "https://external.example"})
        self.assertEqual(status, 403)
        for change in ({"contextStrategy": []}, {"windowSize": False}, {"windowSize": 0}):
            status, _, _ = self.json_request("PUT", "/api/agents/agent-1/settings", {**agent.default_settings(), **change})
            self.assertEqual(status, 400)

    def test_facts_failure_is_visible_and_does_not_send_primary_request(self):
        self.json_request("PUT", "/api/agents/agent-1/settings", {**agent.default_settings(), "contextStrategy": "facts"})
        self.answer = '{"set":{"bad":[]},"delete":[]}'
        status, body, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Запомни"})
        self.assertEqual(status, 502)
        self.assertIn("facts", body["error"])
        self.assertIs(body["accepted"], False)
        self.assertEqual(body["agent"]["context"]["facts"], {})
        self.assertEqual(body["agent"]["context"]["factsUsage"]["calls"], 1)
        self.assertEqual(len(self.calls), 1)

    def test_summary_failure_returns_safe_502(self):
        self.server.registry.update_settings("agent-1", {**agent.default_settings(), "contextStrategy": "summary"})
        for number in range(1, 11):
            self.json_request("POST", "/api/agents/agent-1/messages", {"text": f"Вопрос {number}"})
        agent_instance = self.server.registry.get("agent-1")
        before = agent_instance.snapshot()

        def summary_failure(payload, **options):
            if payload["temperature"] == 0:
                raise RuntimeError("summary provider failure")
            return "Обычный ответ"

        agent_instance._ask_model = summary_failure
        status, body, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Вопрос 11"})

        self.assertEqual(status, 502)
        self.assertEqual(body["error"], "Не удалось обновить сводку истории.")
        self.assertEqual(body["agent"], before)
        self.assertNotIn("summary provider failure", json.dumps(body, ensure_ascii=False))

    def test_missing_key_returns_503_after_saving_message(self):
        self.model_error = errors.MissingApiKeyError("DEEPSEEK_API_KEY")
        status, body, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Привет"})

        self.assertEqual(status, 503)
        self.assertEqual(body["error"], "Не задан DEEPSEEK_API_KEY.")
        self.assertEqual(body["agent"]["messages"], [{"role": "user", "content": "Привет"}])
        self.assertEqual(body["metadata"]["payload"]["model"], agent.MODEL)

    def test_day_three_run_returns_four_methods_and_uses_generated_prompt(self):
        generated_prompt = "Сгенерированный промпт для решения задачи."
        self.answers = [
            "Прямое решение",
            "Пошаговое решение",
            generated_prompt,
            "Решение с использованием сгенерированного промпта",
            "Решение аналитика",
            "Решение инженера",
            "Решение критика",
            "Качественное сравнение ответов",
        ]

        status, body, _ = self.json_request("POST", "/api/day-03/run")

        self.assertEqual(status, 200)
        experiment = body["experiment"]
        self.assertEqual(experiment["taskId"], "server-messages")
        self.assertEqual(experiment["title"], "Три сервера")
        self.assertEqual(experiment["criteria"], list(day_three.DAY_THREE_TASKS["server-messages"]["criteria"]))
        self.assertIn("task", experiment)
        self.assertIn("referenceSolution", experiment)
        self.assertIn("systemPrompt", experiment)
        self.assertEqual(
            [method["id"] for method in experiment["methods"]],
            ["direct", "step_by_step", "generated_prompt", "experts"],
        )
        self.assertEqual([len(method["calls"]) for method in experiment["methods"]], [1, 1, 2, 3])
        generated_method = experiment["methods"][2]
        self.assertEqual(generated_method["calls"][0]["answer"], generated_prompt)
        self.assertEqual(generated_method["calls"][1]["userPrompt"], generated_prompt)
        self.assertEqual(
            generated_method["calls"][1]["answer"],
            "Решение с использованием сгенерированного промпта",
        )
        experts_method = experiment["methods"][3]
        self.assertEqual(
            [call["title"] for call in experts_method["calls"]],
            ["Аналитик", "Инженер", "Критик"],
        )
        self.assertIn("Решение аналитика", experts_method["answer"])
        self.assertIn("Решение инженера", experts_method["answer"])
        self.assertIn("Решение критика", experts_method["answer"])
        self.assertEqual(experiment["comparison"]["status"], "success")
        self.assertEqual(experiment["comparison"]["call"]["answer"], "Качественное сравнение ответов")

        self.assertEqual(len(self.calls), 8)
        self.assertTrue(all(call["payload"]["model"] == agent.MODEL for call in self.calls))
        self.assertEqual(
            [call["payload"]["messages"][0]["content"] for call in self.calls],
            [
                day_three.DIRECT_SYSTEM,
                day_three.STEPWISE_SYSTEM,
                day_three.PROMPT_ENGINEER_SYSTEM,
                day_three.PROMPT_EXECUTOR_SYSTEM,
                day_three.ANALYST_SYSTEM,
                day_three.ENGINEER_SYSTEM,
                day_three.CRITIC_SYSTEM,
                day_three.MODERATOR_SYSTEM,
            ],
        )
        user_prompts = [call["payload"]["messages"][-1]["content"] for call in self.calls]
        self.assertEqual(user_prompts[0], day_three.DAY_THREE_TASKS["server-messages"]["task"])
        self.assertEqual(user_prompts[1], day_three.DAY_THREE_TASKS["server-messages"]["task"])
        self.assertIn(day_three.DAY_THREE_TASKS["server-messages"]["task"], user_prompts[2])
        self.assertEqual(user_prompts[3], generated_prompt)
        self.assertEqual(user_prompts[4], day_three.DAY_THREE_TASKS["server-messages"]["task"])
        self.assertEqual(user_prompts[5], day_three.DAY_THREE_TASKS["server-messages"]["task"])
        self.assertEqual(user_prompts[6], day_three.DAY_THREE_TASKS["server-messages"]["task"])
        self.assertIn("Решение аналитика", user_prompts[7])
        self.assertIn("Решение инженера", user_prompts[7])
        self.assertIn("Решение критика", user_prompts[7])
        self.assertIn(day_three.DAY_THREE_TASKS["server-messages"]["referenceSolution"], user_prompts[7])
        self.assertIn("перебором исключены остальные варианты", user_prompts[7])

    def test_day_three_run_accepts_catalog_task_and_rejects_custom_task(self):
        status, body, _ = self.json_request("POST", "/api/day-03/run", {"taskId": "channel-analysis"})

        self.assertEqual(status, 200)
        self.assertEqual(body["experiment"]["taskId"], "channel-analysis")
        self.assertIn("5 %", body["experiment"]["referenceSolution"])
        self.assertIn("конверс", self.calls[-1]["payload"]["messages"][-1]["content"].lower())

        self.calls.clear()
        status, body, _ = self.json_request("POST", "/api/day-03/run", {"task": "другая задача"})

        self.assertEqual(status, 400)
        self.assertEqual(body, {"error": "Запрос должен содержать только известный taskId."})
        self.assertEqual(self.calls, [])

    def test_day_three_stream_emits_methods_before_comparison(self):
        self.answers = ["direct", "steps", "prompt", "solution", "analyst", "engineer", "critic", "comparison"]
        connection = http.client.HTTPConnection("127.0.0.1", self.port)
        connection.request(
            "POST", "/api/day-03/stream", json.dumps({"taskId": "server-messages"}),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        events = [json.loads(line) for line in response.read().decode("utf-8").splitlines()]
        connection.close()

        self.assertEqual(response.status, 200)
        self.assertEqual(events[0]["type"], "start")
        self.assertEqual({event["method"]["id"] for event in events if event["type"] == "method"}, {
            "direct", "step_by_step", "generated_prompt", "experts",
        })
        self.assertEqual(events[-2]["type"], "comparison")
        self.assertEqual(events[-1]["type"], "complete")

    def test_day_three_catalog_keeps_only_two_fast_tasks(self):
        status, body, _ = self.request("GET", "/")

        self.assertEqual(status, 200)
        self.assertEqual(set(day_three.DAY_THREE_TASKS), {"server-messages", "channel-analysis"})
        self.assertEqual(day_three.DAY_THREE_TASKS["server-messages"]["title"], "Три сервера")
        self.assertEqual(day_three.DAY_THREE_TASKS["channel-analysis"]["title"], "Выбор канала")
        self.assertIn('taskSelect.id="day-three-task-select"', body)
        self.assertIn('id:"server-messages"', body)
        self.assertIn('id:"channel-analysis"', body)
        self.assertNotIn('id:"route-coupon"', body)
        self.assertIn("state.experiments={}", body)
        self.assertIn("activeDayThreeTaskId", body)

    def test_page_keeps_the_stream_line_delimiter_as_javascript_source(self):
        status, body, _ = self.request("GET", "/")

        self.assertEqual(status, 200)
        self.assertIn('buffer.split("\\n")', body)

    def test_day_three_run_rejects_chunked_request_body_without_calling_model(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port)
        connection.request(
            "POST",
            "/api/day-03/run",
            body=[b'{"task":"another"}'],
            headers={"Content-Type": "application/json"},
            encode_chunked=True,
        )
        response = connection.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        connection.close()

        self.assertEqual(response.status, 400)
        self.assertEqual(body, {"error": "Transfer-Encoding не поддерживается."})
        self.assertEqual(self.calls, [])

    def test_day_three_run_rejects_conflicting_content_lengths_without_calling_model(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=1)
        try:
            connection.putrequest("POST", "/api/day-03/run")
            connection.putheader("Content-Length", "0")
            connection.putheader("Content-Length", "2")
            connection.endheaders()
            connection.send(b"{}")
            response = connection.getresponse()
            body = json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()

        self.assertEqual(response.status, 400)
        self.assertEqual(body, {"error": "Неоднозначный Content-Length."})
        self.assertEqual(self.calls, [])

    def test_day_three_run_rejects_invalid_content_length_without_calling_model(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=1)
        try:
            connection.putrequest("POST", "/api/day-03/run")
            connection.putheader("Content-Length", "+0")
            connection.endheaders()
            response = connection.getresponse()
            body = json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()

        self.assertEqual(response.status, 400)
        self.assertEqual(body, {"error": "Некорректный Content-Length."})
        self.assertEqual(self.calls, [])

    def test_day_three_run_provider_error_returns_502_without_partial_experiment(self):
        def failing_model(payload):
            raise RuntimeError("сеть недоступна")

        self.server.ask_model = failing_model
        status, body, _ = self.json_request("POST", "/api/day-03/run")

        self.assertEqual(status, 502)
        self.assertEqual(body, {"error": "Не удалось получить ответы DeepSeek."})
        self.assertNotIn("experiment", body)

    def test_day_three_run_missing_key_returns_503(self):
        with patch.dict(os.environ, {}, clear=True):
            status, body, _ = self.json_request("POST", "/api/day-03/run")

        self.assertEqual(status, 503)
        self.assertEqual(body, {"error": "Не задан DEEPSEEK_API_KEY."})
        self.assertNotIn("experiment", body)

    def test_foreign_origin_is_rejected_for_all_post_routes(self):
        headers = {"Origin": "http://evil.example"}
        for path, payload in [
            ("/api/agents", None),
            ("/api/agents/agent-1/messages", {"text": "Привет"}),
            ("/api/agents/bulk", {"count": 2}),
            ("/api/day-03/run", None),
        ]:
            with self.subTest(path=path):
                status, body, _ = self.json_request("POST", path, payload, headers)
                self.assertEqual(status, 403)
                self.assertEqual(body, {"error": "Запрос с другого источника запрещён."})
        self.assertEqual(self.calls, [])

    def test_memory_api_updates_each_layer_and_persists(self):
        status, body, _ = self.json_request("PUT", "/api/agents/agent-1/memory/working", {
            "task": "Каталог", "data": {"audience": "B2B"},
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["agent"]["context"]["memoryLayers"]["working"]["task"], "Каталог")
        status, body, _ = self.json_request("PUT", "/api/agents/agent-1/memory/long-term/profile", {
            "entries": {"style": "кратко"},
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["agent"]["context"]["memoryLayers"]["longTerm"]["profile"], {"style": "кратко"})

        self.server.server_close()
        self.server = web.ChatServer(("127.0.0.1", 0), self.ask_model, self.state_path)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        status, body, _ = self.json_request("GET", "/api/agents")
        self.assertEqual(body["agents"][0]["context"]["memoryLayers"]["working"]["data"], {"audience": "B2B"})

    def test_memory_api_rejects_bad_body_unknown_agent_and_foreign_origin(self):
        for path, body, expected in [
            ("/api/agents/agent-1/memory/working", {"task": "x"}, 400),
            ("/api/agents/missing/memory/working", {"task": "", "data": {}}, 404),
            ("/api/agents/agent-1/memory/long-term/unknown", {"entries": {}}, 404),
        ]:
            with self.subTest(path=path):
                status, _, _ = self.json_request("PUT", path, body)
                self.assertEqual(status, expected)
        status, body, _ = self.json_request(
            "PUT", "/api/agents/agent-1/memory/working", {"task": "", "data": {}},
            {"Origin": "https://external.example"},
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "Запрос с другого источника запрещён."})

    def test_memory_commands_are_local_persistent_and_clear_only_the_requested_layer(self):
        status, body, _ = self.json_request(
            "POST", "/api/agents/agent-1/messages", {"text": "/working Дизайнерский проект"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["command"]["message"], "Рабочая память сохранена.")
        self.assertEqual(body["agent"]["messages"], [])
        self.assertEqual(body["agent"]["context"]["memoryLayers"]["working"]["task"], "Дизайнерский проект")
        self.assertEqual(self.calls, [])

        status, body, _ = self.json_request(
            "POST", "/api/agents/agent-1/messages", {"text": "/long Меня зовут Диана"},
        )
        self.assertEqual(status, 200)
        self.assertIn("Меня зовут Диана", body["agent"]["context"]["memoryLayers"]["longTerm"]["profile"].values())
        status, body, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "/clear-working"})
        self.assertEqual(status, 200)
        self.assertEqual(body["agent"]["context"]["memoryLayers"]["working"], {"task": "", "data": {}})
        self.assertTrue(body["agent"]["context"]["memoryLayers"]["longTerm"]["profile"])

        self.server.server_close()
        self.server = web.ChatServer(("127.0.0.1", 0), self.ask_model, self.state_path)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        status, body, _ = self.json_request("GET", "/api/agents")
        self.assertTrue(body["agents"][0]["context"]["memoryLayers"]["longTerm"]["profile"])

    def test_memory_commands_reject_incomplete_or_extra_arguments(self):
        for text in ("/working", "/long ", "/clear-long лишнее"):
            with self.subTest(text=text):
                status, body, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": text})
                self.assertEqual(status, 400)
                self.assertTrue(body["error"])
        self.assertEqual(self.calls, [])

    def test_matching_origin_is_accepted(self):
        status, _, _ = self.json_request(
            "POST", "/api/agents/agent-1/messages", {"text": "Привет"},
            {"Origin": f"http://127.0.0.1:{self.port}"},
        )
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
