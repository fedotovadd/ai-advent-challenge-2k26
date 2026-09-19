import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from agent import AgentRegistry, ContextOverflowError, ContextSummaryError, DEFAULT_TEMPERATURE, MODELS, MODEL_CAPABILITIES, OverflowProbeError, PersistenceError
from day_three import (
    DAY_THREE_TASKS,
    MAX_DAY_THREE_REQUEST_BYTES,
    run_day_three_experiment,
    stream_day_three_experiment,
)
from errors import MissingApiKeyError
from context_memory import FactsUpdateError, validate_context_settings
from memory_layers import MemoryCommandError, parse_memory_command


STATIC_PAGE = Path(__file__).with_name("static") / "index.html"
STATIC_AGENT_STATE = Path(__file__).with_name("static") / "agent-state.js"
STATIC_MEMORY_CONTROLS = Path(__file__).with_name("static") / "memory-controls.js"
MAX_MEMORY_REQUEST_BYTES = 65_536


class ChatRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(200, STATIC_PAGE.read_text(encoding="utf-8"))
        elif path == "/static/agent-state.js":
            self._send_javascript(200, STATIC_AGENT_STATE.read_text(encoding="utf-8"))
        elif path == "/static/context-controls.js":
            self._send_javascript(200, STATIC_PAGE.with_name("context-controls.js").read_text(encoding="utf-8"))
        elif path == "/static/memory-controls.js":
            self._send_javascript(200, STATIC_MEMORY_CONTROLS.read_text(encoding="utf-8"))
        elif path == "/api/agents":
            self._send_json(200, {"agents": self.server.registry.agents()})
        elif path.startswith("/api/"):
            self._send_json(404, {"error": "Маршрут не найден."})
        else:
            self._send_html(404, "<h1>Страница не найдена</h1>")

    def do_POST(self):
        if not self._same_origin():
            self._send_json(403, {"error": "Запрос с другого источника запрещён."})
            return
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        if path == "/api/day-03/run":
            self._handle_day_three(parsed_path.query)
            return
        if path == "/api/day-03/stream":
            self._handle_day_three_stream(parsed_path.query)
            return
        if path == "/api/agents":
            try:
                created = self.server.registry.create()
            except PersistenceError:
                self._send_storage_error()
                return
            self._send_json(201, {"agent": created})
            return
        if path == "/api/agents/bulk":
            self._handle_bulk_create()
            return
        parts = path.split("/")
        actions = {"checkpoints": "checkpoint", "branches": "branch", "switch-branch": "switch"}
        if len(parts) == 5 and parts[:3] == ["", "api", "agents"] and parts[4] in actions:
            self._handle_context_action(parts[3], actions[parts[4]])
            return
        if len(parts) == 5 and parts[:3] == ["", "api", "agents"] and parts[4] == "messages":
            self._handle_message(parts[3])
            return
        self._send_json(404, {"error": "Маршрут не найден."})

    def do_PUT(self):
        if not self._same_origin():
            self._send_json(403, {"error": "Запрос с другого источника запрещён."})
            return
        parts = urlparse(self.path).path.split("/")
        if len(parts) == 5 and parts[:3] == ["", "api", "agents"] and parts[4] == "settings":
            self._handle_settings(parts[3])
            return
        if len(parts) == 6 and parts[:3] == ["", "api", "agents"] and parts[4:] == ["memory", "working"]:
            self._handle_memory_update(parts[3], "working")
            return
        if len(parts) == 6 and parts[:3] == ["", "api", "agents"] and parts[4:] == ["memory", "long-term"]:
            self._handle_memory_update(parts[3], "long-term")
            return
        self._send_json(404, {"error": "Маршрут не найден."})

    def do_DELETE(self):
        if not self._same_origin():
            self._send_json(403, {"error": "Запрос с другого источника запрещён."})
            return
        parts = urlparse(self.path).path.split("/")
        if len(parts) != 4 or parts[:3] != ["", "api", "agents"]:
            self._send_json(404, {"error": "Маршрут не найден."})
            return
        agent_id = parts[3]
        try:
            deleted = self.server.registry.delete(agent_id)
        except PersistenceError:
            self._send_storage_error()
            return
        if deleted is None:
            self._send_json(404, {"error": "Агент не найден."})
            return
        self._send_json(200, {"deletedId": agent_id})

    def _same_origin(self):
        origin = self.headers.get("Origin")
        return not origin or origin == f"http://{self.headers.get('Host')}"

    def _handle_day_three(self, query):
        task_id = self._read_day_three_task_id(query)
        if task_id is None:
            return
        if not os.getenv("DEEPSEEK_API_KEY"):
            self._send_json(503, {"error": "Не задан DEEPSEEK_API_KEY."})
            return
        try:
            experiment = run_day_three_experiment(self.server.ask_model, task_id)
        except Exception:
            self._send_json(502, {"error": "Не удалось получить ответы DeepSeek."})
            return
        self._send_json(200, {"experiment": experiment})

    def _handle_day_three_stream(self, query):
        task_id = self._read_day_three_task_id(query)
        if task_id is None:
            return
        if not os.getenv("DEEPSEEK_API_KEY"):
            self._send_json(503, {"error": "Не задан DEEPSEEK_API_KEY."})
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for event in stream_day_three_experiment(self.server.ask_model, task_id):
                self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _read_day_three_task_id(self, query):
        if query:
            self._send_json(400, {"error": "Маршрут не принимает query-параметры."})
            return None
        if self.headers.get("Transfer-Encoding") is not None:
            self._send_json(400, {"error": "Transfer-Encoding не поддерживается."})
            return None
        content_lengths = self.headers.get_all("Content-Length", [])
        if len(content_lengths) > 1:
            self._send_json(400, {"error": "Неоднозначный Content-Length."})
            return None
        if content_lengths and (not content_lengths[0].isascii() or not content_lengths[0].isdigit()):
            self._send_json(400, {"error": "Некорректный Content-Length."})
            return None
        try:
            length = int(content_lengths[0]) if content_lengths else 0
        except ValueError:
            self._send_json(400, {"error": "Некорректный Content-Length."})
            return None
        if length > MAX_DAY_THREE_REQUEST_BYTES:
            self._send_json(400, {"error": "Тело запроса слишком большое."})
            return None
        task_id = "server-messages"
        if length:
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, {"error": "Некорректный JSON."})
                return None
            if not isinstance(data, dict) or set(data) != {"taskId"} or not isinstance(data["taskId"], str):
                self._send_json(400, {"error": "Запрос должен содержать только известный taskId."})
                return None
            task_id = data["taskId"]
            if task_id not in DAY_THREE_TASKS:
                self._send_json(400, {"error": "Запрос должен содержать только известный taskId."})
                return None
        return task_id

    def _handle_bulk_create(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Некорректный JSON."})
            return
        if not isinstance(data, dict) or set(data) != {"count"}:
            self._send_json(400, {"error": "Количество агентов должно быть целым числом от 1 до 100."})
            return
        try:
            agents = self.server.registry.create_many(data["count"])
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return
        except PersistenceError:
            self._send_storage_error()
            return
        self._send_json(201, {"agents": agents})

    def _handle_context_action(self, agent_id, action):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Некорректный размер запроса.")
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.server.registry.context_action(agent_id, action, data)
        except (UnicodeDecodeError, ValueError) as error:
            self._send_json(400, {"error": str(error)})
            return
        except PersistenceError:
            self._send_storage_error()
            return
        if result is None:
            self._send_json(404, {"error": "Агент не найден."})
            return
        self._send_json(200, result)

    def _handle_message(self, agent_id):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Некорректный JSON."})
            return
        if not isinstance(data, dict) or "text" not in data or not isinstance(data["text"], str):
            self._send_json(400, {"error": "Поле text должно быть непустой строкой."})
            return
        try:
            command = parse_memory_command(data["text"])
        except MemoryCommandError as error:
            self._send_json(400, {"error": str(error)})
            return
        if command is not None:
            try:
                result = self.server.registry.apply_memory_command(agent_id, command)
            except MemoryCommandError as error:
                self._send_json(400, {"error": str(error)})
                return
            except PersistenceError:
                self._send_storage_error()
                return
            if result is None:
                self._send_json(404, {"error": "Агент не найден."})
                return
            self._send_json(200, result)
            return
        temperature = data.get("temperature", DEFAULT_TEMPERATURE)
        agent = self.server.registry.get(agent_id)
        if not agent:
            self._send_json(404, {"error": "Агент не найден."})
            return
        try:
            snapshot = self.server.registry.respond(agent_id, data["text"], temperature)
        except PersistenceError:
            self._send_storage_error()
            return
        except MissingApiKeyError as error:
            snapshot = agent.snapshot()
            self._send_json(503, {
                "agent": snapshot,
                "error": f"Не задан {error.key_name}.",
                "accepted": getattr(error, "accepted", True),
                "metadata": snapshot["metadata"],
            })
            return
        except ContextOverflowError as error:
            snapshot = agent.snapshot()
            self._send_json(409, {"agent": snapshot, "error": str(error), "metadata": snapshot["metadata"], "accepted": False})
            return
        except OverflowProbeError as error:
            snapshot = agent.snapshot()
            self._send_json(422, {"agent": snapshot, "error": str(error), "metadata": snapshot["metadata"], "probe": True, "accepted": False})
            return
        except FactsUpdateError as error:
            snapshot = agent.snapshot()
            self._send_json(502, {"agent": snapshot, "error": str(error), "metadata": snapshot["metadata"], "accepted": False})
            return
        except ContextSummaryError:
            snapshot = agent.snapshot()
            self._send_json(502, {
                "agent": snapshot,
                "error": "Не удалось обновить сводку истории.",
                "accepted": False,
                "metadata": snapshot["metadata"],
            })
            return
        except ValueError as error:
            if str(error) in {"Пустое сообщение.", "Поле temperature должно быть числом от 0 до 2."}:
                self._send_json(400, {"error": str(error)})
                return
            snapshot = agent.snapshot()
            self._send_json(502, {
                "agent": snapshot,
                "error": "Не удалось получить ответ DeepSeek.",
                "metadata": snapshot["metadata"],
            })
            return
        except Exception:
            snapshot = agent.snapshot()
            self._send_json(502, {
                "agent": snapshot,
                "error": "Не удалось получить ответ DeepSeek.",
                "metadata": snapshot["metadata"],
            })
            return
        if snapshot is None:
            self._send_json(404, {"error": "Агент не найден."})
            return
        self._send_json(200, {"agent": snapshot, "metadata": snapshot["metadata"]})

    def _handle_settings(self, agent_id):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Некорректный JSON."})
            return
        existing = self.server.registry.get(agent_id)
        settings, error = self._validate_settings(data, existing.snapshot()["settings"] if existing else None)
        if error:
            self._send_json(400, {"error": error})
            return
        agent = self.server.registry.get(agent_id)
        if not agent:
            self._send_json(404, {"error": "Агент не найден."})
            return
        try:
            snapshot = self.server.registry.update_settings(agent_id, settings)
        except PersistenceError:
            self._send_storage_error()
            return
        if snapshot is None:
            self._send_json(404, {"error": "Агент не найден."})
            return
        self._send_json(200, {"agent": snapshot})

    def _read_memory_json_body(self):
        if self.headers.get("Transfer-Encoding") is not None:
            self._send_json(400, {"error": "Transfer-Encoding не поддерживается."})
            return None
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
            self._send_json(400, {"error": "Некорректный Content-Length."})
            return None
        length = int(lengths[0])
        if not 0 < length <= MAX_MEMORY_REQUEST_BYTES:
            self._send_json(400, {"error": "Некорректный размер запроса."})
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Некорректный JSON."})
            return None

    def _handle_memory_update(self, agent_id, scope):
        data = self._read_memory_json_body()
        if data is None:
            return
        entries = data if scope in {"working", "long-term"} else None
        if entries is None:
            self._send_json(400, {"error": "Память имеет неверный формат."})
            return
        try:
            snapshot = self.server.registry.update_memory(agent_id, scope, entries)
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return
        except PersistenceError:
            self._send_storage_error()
            return
        if snapshot is None:
            self._send_json(404, {"error": "Агент не найден."})
            return
        self._send_json(200, {"agent": snapshot})

    def _validate_settings(self, data, current_settings=None):
        legacy_fields = {"model", "systemPrompt", "format", "maxTokens", "stop"}
        day_eight_fields = legacy_fields | {"trainingContextLimit", "sendOnOverflow"}
        compression_fields = {"contextCompressionEnabled"}
        allowed_fields = {
            frozenset(legacy_fields),
            frozenset(day_eight_fields),
            frozenset(legacy_fields | compression_fields),
            frozenset(day_eight_fields | compression_fields),
        }
        strategy_fields = {"contextStrategy", "windowSize"}
        allowed_fields |= {fields | strategy_fields for fields in list(allowed_fields)}
        if not isinstance(data, dict) or frozenset(data) not in allowed_fields:
            return None, "Настройки имеют неверный формат."
        default_strategy = (current_settings or {}).get("contextStrategy", "sliding_window")
        if "contextStrategy" not in data and "contextCompressionEnabled" in data:
            default_strategy = "summary" if data["contextCompressionEnabled"] else "branching"
        data = {
            "trainingContextLimit": None,
            "sendOnOverflow": False,
            "contextCompressionEnabled": True,
            "contextStrategy": default_strategy,
            "windowSize": (current_settings or {}).get("windowSize", 10),
            **data,
        }
        system_prompt = data["systemPrompt"]
        model = data["model"]
        response_format = data["format"]
        max_tokens = data["maxTokens"]
        stop = data["stop"]
        training_limit = data["trainingContextLimit"]
        send_on_overflow = data["sendOnOverflow"]
        compression_enabled = data["contextCompressionEnabled"]
        try:
            validate_context_settings(data)
        except ValueError as error:
            return None, str(error)
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            return None, "System prompt не может быть пустым."
        if not isinstance(model, str) or model not in MODELS:
            return None, "Неизвестная модель."
        if not isinstance(response_format, str) or response_format not in {"text", "json"}:
            return None, "Формат ответа должен быть text или json."
        if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0 or max_tokens > MODEL_CAPABILITIES[model]["maxOutputTokens"]):
            return None, "Максимум токенов должен быть положительным целым числом."
        if not isinstance(stop, str):
            return None, "Стоп-последовательность должна быть строкой."
        if training_limit is not None and (isinstance(training_limit, bool) or not isinstance(training_limit, int) or training_limit <= 0):
            return None, "Учебный лимит контекста должен быть положительным целым числом."
        if not isinstance(send_on_overflow, bool):
            return None, "Флаг демонстрации переполнения должен быть логическим значением."
        if not isinstance(compression_enabled, bool):
            return None, "Флаг сжатия контекста должен быть логическим значением."
        return {
            "model": model,
            "systemPrompt": system_prompt.strip(),
            "format": response_format,
            "maxTokens": max_tokens,
            "stop": stop.strip(),
            "trainingContextLimit": training_limit,
            "sendOnOverflow": send_on_overflow,
            "contextCompressionEnabled": compression_enabled,
            "contextStrategy": data["contextStrategy"],
            "windowSize": data["windowSize"],
        }, None

    def _send_json(self, status, body):
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_storage_error(self):
        self._send_json(500, {"error": "Не удалось сохранить состояние агента."})

    def _send_html(self, status, body):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_javascript(self, status, body):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/javascript; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class ChatServer(ThreadingHTTPServer):
    def __init__(self, address, ask_model, state_path=None):
        super().__init__(address, ChatRequestHandler)
        self.registry = AgentRegistry(ask_model, state_path)
        self.ask_model = ask_model
