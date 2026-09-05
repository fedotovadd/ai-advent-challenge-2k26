import http.client
import json
import os
import threading
import unittest
from unittest.mock import patch

import main


class DeepSeekWebTests(unittest.TestCase):
    def setUp(self):
        self.payloads = []
        self.answer = "Тестовый ответ"
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

    def ask_model(self, payload):
        self.payloads.append(payload)
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

    def test_page_contains_chat_controls(self):
        status, body, headers = self.request("GET", "/")

        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn("Новый чат", body)
        self.assertIn("Метаданные", body)
        self.assertIn("message-input", body)
        self.assertNotIn("DEEPSEEK_API_KEY", body)

    def test_initial_session_is_available(self):
        status, body, headers = self.json_request("GET", "/api/sessions")

        self.assertEqual(status, 200)
        self.assertIn("application/json", headers["Content-Type"])
        self.assertEqual(
            body,
            {"sessions": [{"id": "session-1", "title": "Новый чат", "messages": []}]},
        )

    def test_create_session_assigns_next_id(self):
        status, body, _ = self.json_request("POST", "/api/sessions")

        self.assertEqual(status, 201)
        self.assertEqual(body["session"], {"id": "session-2", "title": "Новый чат", "messages": []})

    def test_message_returns_answer_and_exact_metadata(self):
        status, body, _ = self.json_request("POST", "/api/sessions/session-1/messages", {"text": "  Привет  "})

        expected_messages = [
            {"role": "system", "content": main.SYSTEM_PROMPT},
            {"role": "user", "content": "Привет"},
        ]
        self.assertEqual(status, 200)
        self.assertEqual(body["session"], {
            "id": "session-1", "title": "Привет",
            "messages": [{"role": "user", "content": "Привет"}, {"role": "assistant", "content": "Тестовый ответ"}],
        })
        self.assertEqual(self.payloads, [{"model": main.MODEL, "messages": expected_messages}])
        self.assertEqual(body["metadata"], {
            "userPrompt": "Привет", "systemPrompt": main.SYSTEM_PROMPT,
            "payload": {"model": main.MODEL, "messages": expected_messages},
            "status": {"kind": "success", "label": "200 OK"},
        })

    def test_second_message_sends_complete_history(self):
        self.json_request("POST", "/api/sessions/session-1/messages", {"text": "Первый"})
        self.answer = "Второй ответ"
        status, _, _ = self.json_request("POST", "/api/sessions/session-1/messages", {"text": "Второй"})

        self.assertEqual(status, 200)
        self.assertEqual(self.payloads[-1]["messages"], [
            {"role": "system", "content": main.SYSTEM_PROMPT},
            {"role": "user", "content": "Первый"},
            {"role": "assistant", "content": "Тестовый ответ"},
            {"role": "user", "content": "Второй"},
        ])

    def test_first_message_title_is_limited_to_forty_characters(self):
        text = "а" * 45
        status, body, _ = self.json_request("POST", "/api/sessions/session-1/messages", {"text": text})

        self.assertEqual(status, 200)
        self.assertEqual(body["session"]["title"], "а" * 40)

    def test_invalid_message_bodies_are_rejected(self):
        cases = [
            ({"text": "   "}, "Пустое сообщение."),
            ({}, "Поле text должно быть непустой строкой."),
            ({"text": 12}, "Поле text должно быть непустой строкой."),
        ]
        for payload, expected_error in cases:
            with self.subTest(payload=payload):
                status, body, _ = self.json_request("POST", "/api/sessions/session-1/messages", payload)
                self.assertEqual(status, 400)
                self.assertEqual(body, {"error": expected_error})

        status, body, _ = self.request("POST", "/api/sessions/session-1/messages", headers={"Content-Type": "application/json"})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body), {"error": "Некорректный JSON."})

    def test_unknown_session_returns_json_404(self):
        status, body, _ = self.json_request("POST", "/api/sessions/unknown/messages", {"text": "Привет"})

        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "Сессия не найдена."})

    def test_provider_error_preserves_session_and_metadata(self):
        def failing_model(payload):
            raise RuntimeError("сеть недоступна")

        self.server.ask_model = failing_model
        status, body, _ = self.json_request("POST", "/api/sessions/session-1/messages", {"text": "Привет"})

        self.assertEqual(status, 502)
        self.assertEqual(body["error"], "Не удалось получить ответ DeepSeek.")
        self.assertEqual(body["session"]["messages"], [{"role": "user", "content": "Привет"}])
        self.assertEqual(body["metadata"]["status"], {"kind": "error", "label": "Ошибка API"})
        self.assertNotIn("сеть недоступна", json.dumps(body, ensure_ascii=False))

    def test_empty_provider_answer_returns_502(self):
        self.answer = ""
        status, body, _ = self.json_request("POST", "/api/sessions/session-1/messages", {"text": "Привет"})

        self.assertEqual(status, 502)
        self.assertEqual(body["error"], "Не удалось получить ответ DeepSeek.")
        self.assertEqual(body["session"]["messages"], [{"role": "user", "content": "Привет"}])

    def test_missing_key_returns_503_after_saving_message(self):
        with patch.dict(os.environ, {}, clear=True):
            status, body, _ = self.json_request("POST", "/api/sessions/session-1/messages", {"text": "Привет"})

        self.assertEqual(status, 503)
        self.assertEqual(body["error"], "Не задан DEEPSEEK_API_KEY.")
        self.assertEqual(body["session"]["messages"], [{"role": "user", "content": "Привет"}])
        self.assertEqual(body["metadata"]["payload"]["model"], main.MODEL)

    def test_foreign_origin_is_rejected_for_all_post_routes(self):
        headers = {"Origin": "http://evil.example"}
        for path, payload in [("/api/sessions", None), ("/api/sessions/session-1/messages", {"text": "Привет"})]:
            with self.subTest(path=path):
                status, body, _ = self.json_request("POST", path, payload, headers)
                self.assertEqual(status, 403)
                self.assertEqual(body, {"error": "Запрос с другого источника запрещён."})
        self.assertEqual(self.payloads, [])

    def test_matching_origin_is_accepted(self):
        status, _, _ = self.json_request(
            "POST", "/api/sessions/session-1/messages", {"text": "Привет"},
            {"Origin": f"http://127.0.0.1:{self.port}"},
        )
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
