import copy
import json
import math
import os
import threading
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from errors import MissingApiKeyError


MODEL = "deepseek-v4-flash"
MODELS = ("deepseek-v4-flash", "glm-4.7-flash", "deepseek-v4-pro")
MODEL_PRICING = {
    "deepseek-v4-flash": {"input": 0.22, "output": 0.66},
    "glm-4.7-flash": None,
    "deepseek-v4-pro": {"input": 0.66, "output": 1.98},
}
SYSTEM_PROMPT = (
    "Ты AI-помощник. Отвечай ясно, кратко, по-русски. "
    "Возвращай обычный текст без Markdown-разметки."
)
JSON_OUTPUT_INSTRUCTION = "Верни только валидный JSON без Markdown-разметки."
DEFAULT_TEMPERATURE = 1
DEFAULT_SETTINGS = {
    "model": MODEL,
    "systemPrompt": SYSTEM_PROMPT,
    "format": "text",
    "maxTokens": None,
    "stop": "",
}
MAX_BULK_AGENTS = 100
STATE_VERSION = 1
DEFAULT_STATE_PATH = Path(__file__).with_name("data") / "agents.json"
METADATA_FIELDS = {
    "userPrompt",
    "systemPrompt",
    "payload",
    "status",
    "responseTimeMs",
    "usage",
    "cost",
}


class PersistenceError(Exception):
    pass


def default_settings():
    return copy.deepcopy(DEFAULT_SETTINGS)


def _response_content_and_usage(response):
    if isinstance(response, str):
        return response, None
    if isinstance(response, dict):
        return response.get("content"), response.get("usage")
    return None, None


def _usage_value(usage, name):
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def request_usage_and_cost(model, usage):
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    details = None if usage is None else {
        "promptTokens": prompt_tokens,
        "completionTokens": completion_tokens,
        "totalTokens": total_tokens,
    }
    pricing = MODEL_PRICING[model]
    if pricing is None:
        return details, {"kind": "free"}
    if prompt_tokens is None or completion_tokens is None:
        return details, None
    return details, {
        "kind": "paid",
        "usd": round((prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000, 12),
    }


class Agent:
    def __init__(self, agent_id, name, settings, ask_model, messages=None, metadata=None):
        self._lock = threading.RLock()
        self._id = agent_id
        self._name = name
        self._settings = copy.deepcopy(settings)
        self._ask_model = ask_model
        self._messages = copy.deepcopy(messages) if messages is not None else []
        self._metadata = copy.deepcopy(metadata)

    def snapshot(self):
        with self._lock:
            return {
                "id": self._id,
                "name": self._name,
                "messages": copy.deepcopy(self._messages),
                "settings": copy.deepcopy(self._settings),
                "metadata": copy.deepcopy(self._metadata),
            }

    def update_settings(self, settings):
        with self._lock:
            self._settings = copy.deepcopy(settings)
            return self.snapshot()

    def respond(self, text, temperature=DEFAULT_TEMPERATURE):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Пустое сообщение.")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise ValueError("Поле temperature должно быть числом от 0 до 2.")

        with self._lock:
            text = text.strip()
            self._messages.append({"role": "user", "content": text})
            system_prompt = self._settings["systemPrompt"]
            options = {}
            if self._settings["format"] == "json":
                system_prompt = f"{system_prompt}\n\n{JSON_OUTPUT_INSTRUCTION}"
                options["response_format"] = {"type": "json_object"}
            if self._settings["maxTokens"] is not None:
                options["max_tokens"] = self._settings["maxTokens"]
            if self._settings["stop"]:
                options["stop"] = self._settings["stop"]
            payload = {
                "model": self._settings["model"],
                "messages": [{"role": "system", "content": system_prompt}, *self._messages],
                "temperature": temperature,
            }
            metadata = {
                "userPrompt": text,
                "systemPrompt": system_prompt,
                "payload": {**payload, **options},
                "status": {"kind": "success", "label": "200 OK"},
                "responseTimeMs": None,
                "usage": None,
                "cost": None,
            }
            try:
                started_at = time.monotonic()
                answer, usage = _response_content_and_usage(self._ask_model(payload, **options))
                metadata["responseTimeMs"] = round((time.monotonic() - started_at) * 1000)
                metadata["usage"], metadata["cost"] = request_usage_and_cost(self._settings["model"], usage)
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("empty API response")
            except Exception:
                metadata["status"] = {"kind": "error", "label": "Ошибка API"}
                self._metadata = metadata
                raise
            self._metadata = metadata
            self._messages.append({"role": "assistant", "content": answer})
            return self.snapshot()


def _agent_number(agent_id):
    if not isinstance(agent_id, str) or not agent_id.startswith("agent-"):
        return None
    number = agent_id[6:]
    if not number.isascii() or not number.isdigit() or number.startswith("0"):
        return None
    return int(number) if int(number) > 0 else None


def _valid_settings(settings):
    if not isinstance(settings, dict) or set(settings) != set(DEFAULT_SETTINGS):
        return False
    max_tokens = settings["maxTokens"]
    return (
        settings["model"] in MODELS
        and isinstance(settings["systemPrompt"], str)
        and bool(settings["systemPrompt"].strip())
        and settings["format"] in {"text", "json"}
        and (
            max_tokens is None
            or (isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > 0)
        )
        and isinstance(settings["stop"], str)
    )


def _valid_message(message):
    return (
        isinstance(message, dict)
        and set(message) == {"role", "content"}
        and message["role"] in {"user", "assistant"}
        and isinstance(message["content"], str)
        and bool(message["content"].strip())
    )


def _valid_metadata(metadata):
    if metadata is None:
        return True
    response_time = metadata.get("responseTimeMs") if isinstance(metadata, dict) else None
    status = metadata.get("status") if isinstance(metadata, dict) else None
    return (
        isinstance(metadata, dict)
        and set(metadata) == METADATA_FIELDS
        and isinstance(metadata["userPrompt"], str)
        and isinstance(metadata["systemPrompt"], str)
        and isinstance(metadata["payload"], dict)
        and isinstance(status, dict)
        and set(status) == {"kind", "label"}
        and isinstance(status["kind"], str)
        and isinstance(status["label"], str)
        and (
            response_time is None
            or (isinstance(response_time, int) and not isinstance(response_time, bool) and response_time >= 0)
        )
        and (metadata["usage"] is None or isinstance(metadata["usage"], dict))
        and (metadata["cost"] is None or isinstance(metadata["cost"], dict))
    )


def _valid_agent_snapshot(snapshot):
    return (
        isinstance(snapshot, dict)
        and set(snapshot) == {"id", "name", "messages", "settings", "metadata"}
        and _agent_number(snapshot["id"]) is not None
        and isinstance(snapshot["name"], str)
        and bool(snapshot["name"].strip())
        and isinstance(snapshot["messages"], list)
        and all(_valid_message(message) for message in snapshot["messages"])
        and _valid_settings(snapshot["settings"])
        and _valid_metadata(snapshot["metadata"])
    )


class AgentRegistry:
    def __init__(self, ask_model, state_path=None):
        self._lock = threading.RLock()
        self._ask_model = ask_model
        self._state_path = Path(state_path) if state_path is not None else DEFAULT_STATE_PATH
        restored = self._load()
        if restored is None:
            self._agents = {
                "agent-1": Agent("agent-1", "Агент 1", default_settings(), ask_model),
            }
            self._next_id = 2
        else:
            self._agents, self._next_id = restored

    def agents(self):
        with self._lock:
            return [agent.snapshot() for agent in self._agents.values()]

    def get(self, agent_id):
        with self._lock:
            return self._agents.get(agent_id)

    def create(self):
        with self._lock:
            agent_id = f"agent-{self._next_id}"
            self._next_id += 1
            agent = Agent(agent_id, f"Агент {agent_id[6:]}", default_settings(), self._ask_model)
            self._agents[agent_id] = agent
            snapshot = agent.snapshot()
            self._save()
            return snapshot

    def create_many(self, count):
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_BULK_AGENTS:
            raise ValueError("Количество агентов должно быть целым числом от 1 до 100.")
        with self._lock:
            created = []
            for _ in range(count):
                agent_number = self._next_id
                agent_id = f"agent-{agent_number}"
                agent = Agent(agent_id, f"Агент {agent_number}", default_settings(), self._ask_model)
                self._agents[agent_id] = agent
                self._next_id += 1
                created.append(agent.snapshot())
            self._save()
            return created

    def update_settings(self, agent_id, settings):
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                return None
            snapshot = agent.update_settings(settings)
            self._save()
            return snapshot

    def respond(self, agent_id, text, temperature=DEFAULT_TEMPERATURE):
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                return None
            try:
                snapshot = agent.respond(text, temperature)
            except Exception:
                self._save()
                raise
            self._save()
            return snapshot

    def _state(self):
        with self._lock:
            return {
                "version": STATE_VERSION,
                "nextId": self._next_id,
                "agents": [agent.snapshot() for agent in self._agents.values()],
            }

    def _save(self):
        temporary_path = None
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                "w", encoding="utf-8", dir=self._state_path.parent, delete=False,
            ) as temporary_file:
                json.dump(self._state(), temporary_file, ensure_ascii=False)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = Path(temporary_file.name)
            os.replace(temporary_path, self._state_path)
        except (OSError, TypeError, ValueError) as error:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise PersistenceError("Не удалось сохранить состояние агента.") from error

    def _load(self):
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if (
            not isinstance(state, dict)
            or set(state) != {"version", "nextId", "agents"}
            or state["version"] != STATE_VERSION
            or not isinstance(state["nextId"], int)
            or isinstance(state["nextId"], bool)
            or not isinstance(state["agents"], list)
            or not state["agents"]
            or not all(_valid_agent_snapshot(snapshot) for snapshot in state["agents"])
        ):
            return None
        ids = [snapshot["id"] for snapshot in state["agents"]]
        numbers = [_agent_number(agent_id) for agent_id in ids]
        if len(set(ids)) != len(ids) or state["nextId"] <= max(numbers):
            return None
        return {
            snapshot["id"]: Agent(
                snapshot["id"],
                snapshot["name"],
                snapshot["settings"],
                self._ask_model,
                snapshot["messages"],
                snapshot["metadata"],
            )
            for snapshot in state["agents"]
        }, state["nextId"]
