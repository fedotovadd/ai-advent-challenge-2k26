import http.client
import json
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main


class DeepSeekWebTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.answers = []
        self.answer = "Тестовый ответ"
        self.model_error = None
        self.environment = patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"})
        self.environment.start()
        self.server = main.create_server("127.0.0.1", 0, self.ask_model)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.environment.stop()

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
        self.assertIn('id="agent-count-input"', body)
        self.assertIn('id="create-many-agents"', body)
        self.assertIn('id="agent-list"', body)
        self.assertIn("Метаданные", body)
        self.assertIn("message-input", body)
        self.assertNotIn("DEEPSEEK_API_KEY", body)

    @patch("main.OpenAI")
    def test_glm_model_uses_zai_endpoint_and_key(self, openai):
        openai.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Ответ GLM"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        payload = {
            "model": "glm-4.7-flash",
            "messages": [{"role": "user", "content": "Привет"}],
            "temperature": 1,
        }

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "deepseek-key", "ZAI_API_KEY": "zai-key"}, clear=True):
            result = main.ask_deepseek(payload)

        openai.assert_called_once_with(api_key="zai-key", base_url="https://api.z.ai/api/paas/v4")
        self.assertEqual(result["content"], "Ответ GLM")

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

    def test_page_contains_model_and_usage_metadata(self):
        status, body, _ = self.request("GET", "/")

        self.assertEqual(status, 200)
        self.assertIn("model-input", body)
        self.assertIn("Время ответа", body)
        self.assertIn("Токены", body)
        self.assertIn("Стоимость запроса", body)

    def test_agent_switch_handler_is_not_rendered_as_agent_label(self):
        status, body, _ = self.request("GET", "/")

        self.assertEqual(status, 200)
        self.assertIn("button.append(title,detail);", body)
        self.assertIn('button.addEventListener("click",async()=>', body)
        self.assertIn("/api/agents", body)
        self.assertIn("/api/agents/bulk", body)
        self.assertNotIn("/api/sessions", body)

    def test_initial_agent_is_available(self):
        status, body, headers = self.json_request("GET", "/api/agents")

        self.assertEqual(status, 200)
        self.assertIn("application/json", headers["Content-Type"])
        self.assertEqual(
            body,
            {"agents": [{"id": "agent-1", "name": "Первый агент", "messages": [], "settings": {
                "model": main.MODEL,
                "systemPrompt": main.SYSTEM_PROMPT,
                "format": "text",
                "maxTokens": None,
                "stop": "",
            }, "metadata": None}]},
        )

    def test_create_agent_assigns_next_id(self):
        status, body, _ = self.json_request("POST", "/api/agents")

        self.assertEqual(status, 201)
        self.assertEqual(body["agent"], {"id": "agent-2", "name": "Агент 2", "messages": [], "settings": {
            "model": main.MODEL,
            "systemPrompt": main.SYSTEM_PROMPT,
            "format": "text",
            "maxTokens": None,
            "stop": "",
        }, "metadata": None})

    def test_bulk_create_returns_requested_independent_agents(self):
        status, body, _ = self.json_request("POST", "/api/agents/bulk", {"count": 3})

        self.assertEqual(status, 201)
        self.assertEqual([agent["id"] for agent in body["agents"]], ["agent-2", "agent-3", "agent-4"])
        self.assertEqual(len(self.server.registry.agents()), 4)
        self.assertEqual(len({
            (agent["name"], agent["settings"]["model"], agent["settings"]["systemPrompt"])
            for agent in body["agents"]
        }), 3)
        self.assertEqual(self.calls, [])

    def test_bulk_create_rejects_invalid_count(self):
        for payload in ({}, {"count": 0}, {"count": 101}, {"count": True}, {"count": "3"}):
            with self.subTest(payload=payload):
                status, body, _ = self.json_request("POST", "/api/agents/bulk", payload)
                self.assertEqual(status, 400)
                self.assertIn("количество", body["error"].lower())

    def test_settings_are_stored_per_agent(self):
        settings = {
            "model": main.MODEL,
            "systemPrompt": "Отвечай кратко.",
            "format": "json",
            "maxTokens": 300,
            "stop": "<END>",
        }

        status, body, _ = self.json_request("PUT", "/api/agents/agent-1/settings", settings)

        self.assertEqual(status, 200)
        self.assertEqual(body["agent"]["settings"], settings)

    def test_agent_model_and_metadata_are_persisted(self):
        settings = {
            "systemPrompt": main.SYSTEM_PROMPT,
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
            "promptTokens": 100, "completionTokens": 50, "totalTokens": 150,
        })
        self.assertEqual(body["metadata"]["cost"], {"kind": "paid", "usd": 0.000165})
        self.assertGreaterEqual(body["metadata"]["responseTimeMs"], 0)
        self.assertEqual(agents["agents"][0]["metadata"], body["metadata"])

    def test_invalid_settings_do_not_replace_saved_values(self):
        valid_settings = {
            "model": main.MODEL,
            "systemPrompt": "Отвечай кратко.",
            "format": "text",
            "maxTokens": None,
            "stop": "",
        }
        self.json_request("PUT", "/api/agents/agent-1/settings", valid_settings)

        status, body, _ = self.json_request(
            "PUT", "/api/agents/agent-1/settings",
            {"model": main.MODEL, "systemPrompt": "   ", "format": "text", "maxTokens": 0, "stop": ""},
        )

        self.assertEqual(status, 400)
        self.assertIn("system prompt", body["error"].lower())
        _, agents, _ = self.json_request("GET", "/api/agents")
        self.assertEqual(agents["agents"][0]["settings"], valid_settings)

    def test_settings_for_unknown_agent_return_404(self):
        status, body, _ = self.json_request(
            "PUT", "/api/agents/missing/settings",
            {"model": main.MODEL, "systemPrompt": "Отвечай кратко.", "format": "text", "maxTokens": None, "stop": ""},
        )

        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "Агент не найден."})

    def test_legacy_format_instruction_setting_is_rejected(self):
        valid_settings = {
            "model": main.MODEL,
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
                "model": main.MODEL,
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
        self.assertEqual(agents["agents"][0]["settings"], valid_settings)

    def test_message_returns_answer_and_exact_metadata(self):
        status, body, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "  Привет  "})

        expected_messages = [
            {"role": "system", "content": main.SYSTEM_PROMPT},
            {"role": "user", "content": "Привет"},
        ]
        self.assertEqual(status, 200)
        self.assertEqual(body["agent"]["id"], "agent-1")
        self.assertEqual(body["agent"]["name"], "Первый агент")
        self.assertEqual(body["agent"]["messages"], [
            {"role": "user", "content": "Привет"}, {"role": "assistant", "content": "Тестовый ответ"},
        ])
        self.assertEqual(body["agent"]["settings"]["model"], main.MODEL)
        self.assertEqual(body["agent"]["metadata"], body["metadata"])
        self.assertEqual(self.calls, [{"payload": {
            "model": main.MODEL,
            "messages": expected_messages,
            "temperature": 1,
        }, "kwargs": {}}])
        self.assertEqual(body["metadata"]["userPrompt"], "Привет")
        self.assertEqual(body["metadata"]["systemPrompt"], main.SYSTEM_PROMPT)
        self.assertEqual(body["metadata"]["payload"], {
            "model": main.MODEL, "messages": expected_messages, "temperature": 1,
        })
        self.assertEqual(body["metadata"]["status"], {"kind": "success", "label": "200 OK"})
        self.assertIsNone(body["metadata"]["usage"])
        self.assertIsNone(body["metadata"]["cost"])

    def test_second_message_sends_complete_history(self):
        self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Первый"})
        self.answer = "Второй ответ"
        status, _, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Второй"})

        self.assertEqual(status, 200)
        self.assertEqual(self.calls[-1]["payload"]["messages"], [
            {"role": "system", "content": main.SYSTEM_PROMPT},
            {"role": "user", "content": "Первый"},
            {"role": "assistant", "content": "Тестовый ответ"},
            {"role": "user", "content": "Второй"},
        ])

    def test_json_settings_are_sent_to_the_model(self):
        settings = {
            "model": main.MODEL,
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
            "model": main.MODEL,
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

    def test_missing_key_returns_503_after_saving_message(self):
        self.model_error = main.MissingApiKeyError("DEEPSEEK_API_KEY")
        status, body, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Привет"})

        self.assertEqual(status, 503)
        self.assertEqual(body["error"], "Не задан DEEPSEEK_API_KEY.")
        self.assertEqual(body["agent"]["messages"], [{"role": "user", "content": "Привет"}])
        self.assertEqual(body["metadata"]["payload"]["model"], main.MODEL)

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
        self.assertEqual(experiment["criteria"], list(main.DAY_THREE_TASKS["server-messages"]["criteria"]))
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
        self.assertTrue(all(call["payload"]["model"] == main.MODEL for call in self.calls))
        self.assertEqual(
            [call["payload"]["messages"][0]["content"] for call in self.calls],
            [
                main.DIRECT_SYSTEM,
                main.STEPWISE_SYSTEM,
                main.PROMPT_ENGINEER_SYSTEM,
                main.PROMPT_EXECUTOR_SYSTEM,
                main.ANALYST_SYSTEM,
                main.ENGINEER_SYSTEM,
                main.CRITIC_SYSTEM,
                main.MODERATOR_SYSTEM,
            ],
        )
        user_prompts = [call["payload"]["messages"][-1]["content"] for call in self.calls]
        self.assertEqual(user_prompts[0], main.DAY_THREE_TASKS["server-messages"]["task"])
        self.assertEqual(user_prompts[1], main.DAY_THREE_TASKS["server-messages"]["task"])
        self.assertIn(main.DAY_THREE_TASKS["server-messages"]["task"], user_prompts[2])
        self.assertEqual(user_prompts[3], generated_prompt)
        self.assertEqual(user_prompts[4], main.DAY_THREE_TASKS["server-messages"]["task"])
        self.assertEqual(user_prompts[5], main.DAY_THREE_TASKS["server-messages"]["task"])
        self.assertEqual(user_prompts[6], main.DAY_THREE_TASKS["server-messages"]["task"])
        self.assertIn("Решение аналитика", user_prompts[7])
        self.assertIn("Решение инженера", user_prompts[7])
        self.assertIn("Решение критика", user_prompts[7])
        self.assertIn(main.DAY_THREE_TASKS["server-messages"]["referenceSolution"], user_prompts[7])
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
        self.assertEqual(set(main.DAY_THREE_TASKS), {"server-messages", "channel-analysis"})
        self.assertEqual(main.DAY_THREE_TASKS["server-messages"]["title"], "Три сервера")
        self.assertEqual(main.DAY_THREE_TASKS["channel-analysis"]["title"], "Выбор канала")
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

    def test_matching_origin_is_accepted(self):
        status, _, _ = self.json_request(
            "POST", "/api/agents/agent-1/messages", {"text": "Привет"},
            {"Origin": f"http://127.0.0.1:{self.port}"},
        )
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
