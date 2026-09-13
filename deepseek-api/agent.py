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
MODEL_CAPABILITIES = {
    "deepseek-v4-flash": {"contextLimit": 1_000_000, "maxOutputTokens": 384_000, "safeOutputTokens": 4_096},
    "deepseek-v4-pro": {"contextLimit": 1_000_000, "maxOutputTokens": 384_000, "safeOutputTokens": 4_096},
    "glm-4.7-flash": {"contextLimit": 200_000, "maxOutputTokens": 131_072, "safeOutputTokens": 4_096},
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
    "trainingContextLimit": None,
    "sendOnOverflow": False,
    "contextCompressionEnabled": True,
}
MAX_BULK_AGENTS = 100
STATE_VERSION = 3
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


class ContextOverflowError(ValueError):
    pass


class OverflowProbeError(RuntimeError):
    pass


def default_settings():
    return copy.deepcopy(DEFAULT_SETTINGS)


def default_metrics():
    return {
        "calls": [],
        "totals": {
            "promptTokens": 0,
            "completionTokens": 0,
            "totalTokens": 0,
            "usd": 0,
            "compressionGrossSavedTokens": 0,
            "summaryCallTokens": 0,
            "compressionNetSavedTokens": 0,
            "grossInputSavingsUsd": 0,
            "summaryCostUsd": 0,
            "netSavingsUsd": 0,
        },
        "lastMessage": None,
        "lastContextAttempt": None,
    }


def default_context():
    return {
        "summary": None,
        "compressedMessageCount": 0,
        "summaryUsage": {
            "calls": 0,
            "promptTokens": 0,
            "completionTokens": 0,
            "totalTokens": 0,
            "usd": 0,
            "last": None,
        },
    }


def _response_content_and_usage(response):
    if isinstance(response, str):
        return response, None
    if isinstance(response, dict):
        return response.get("content"), response.get("usage")
    return None, None


def provider_error_message(error):
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        message = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else None
        message = message or body.get("message")
        if isinstance(message, str) and message:
            return message
    message = getattr(error, "message", None)
    return message if isinstance(message, str) and message else str(error)


def _usage_value(usage, name):
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def estimate_text_tokens(text):
    """A stable, deliberately approximate estimate suitable for local UI feedback."""
    if not isinstance(text, str) or not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def estimate_payload_tokens(payload):
    messages = payload.get("messages", []) if isinstance(payload, dict) else []
    message_tokens = sum(
        4 + estimate_text_tokens(message.get("role", "")) + estimate_text_tokens(message.get("content", ""))
        for message in messages if isinstance(message, dict)
    )
    option_tokens = sum(estimate_text_tokens(str(value)) for key, value in payload.items() if key != "messages") if isinstance(payload, dict) else 0
    return 3 + message_tokens + option_tokens


def request_usage_and_cost(model, usage, estimated_prompt_tokens=None, estimated_completion_tokens=None):
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    actual = prompt_tokens is not None and completion_tokens is not None
    if actual and total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    if not actual:
        prompt_tokens = estimated_prompt_tokens
        completion_tokens = estimated_completion_tokens
        total_tokens = (prompt_tokens + completion_tokens
                        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int) else None)
    details = None if total_tokens is None else {
        "promptTokens": prompt_tokens, "completionTokens": completion_tokens,
        "totalTokens": total_tokens, "source": "actual" if actual else "estimated",
    }
    pricing = MODEL_PRICING[model]
    if pricing is None:
        return details, {"kind": "free", "usd": 0, "source": details["source"] if details else "estimated"}
    if prompt_tokens is None or completion_tokens is None:
        return details, None
    return details, {
        "kind": "paid",
        "usd": (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000,
        "source": details["source"],
    }


class Agent:
    def __init__(self, agent_id, name, settings, ask_model, messages=None, metadata=None, metrics=None, context=None):
        self._lock = threading.RLock()
        self._id = agent_id
        self._name = name
        self._settings = {**default_settings(), **copy.deepcopy(settings)}
        self._ask_model = ask_model
        self._messages = copy.deepcopy(messages) if messages is not None else []
        self._metadata = copy.deepcopy(metadata)
        self._metrics = copy.deepcopy(metrics) if metrics is not None else default_metrics()
        self._context = copy.deepcopy(context) if context is not None else default_context()

    def snapshot(self):
        with self._lock:
            return {
                "id": self._id,
                "name": self._name,
                "messages": copy.deepcopy(self._messages),
                "settings": copy.deepcopy(self._settings),
                "metadata": copy.deepcopy(self._metadata),
                "metrics": copy.deepcopy(self._metrics),
                "context": copy.deepcopy(self._context),
            }

    def update_settings(self, settings):
        with self._lock:
            self._settings = {**default_settings(), **copy.deepcopy(settings)}
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
                "messages": [{"role": "system", "content": system_prompt}, *self._messages, {"role": "user", "content": text}],
                "temperature": temperature,
            }
            full_payload = {**payload, **options}
            estimated_payload = estimate_payload_tokens(full_payload)
            capability = MODEL_CAPABILITIES[self._settings["model"]]
            context_limit = self._settings["trainingContextLimit"] or capability["contextLimit"]
            requested_output = self._settings["maxTokens"] or capability["safeOutputTokens"]
            context_percent = estimated_payload / context_limit
            overflow = estimated_payload + requested_output > context_limit
            if overflow:
                self._metrics["lastContextAttempt"] = {
                    "payloadEstimatedTokens": estimated_payload,
                    "requestedOutputTokens": requested_output,
                    "contextLimit": context_limit,
                    "contextPercent": context_percent,
                    "reason": "История и возможный ответ превышают лимит контекста модели.",
                    "mode": "probe" if self._settings["sendOnOverflow"] else "blocked",
                }
                if not self._settings["sendOnOverflow"]:
                    raise ContextOverflowError("Сообщение не отправлено: история и возможный ответ превышают лимит контекста модели.")
            metadata = {
                "userPrompt": text,
                "systemPrompt": system_prompt,
                "payload": full_payload,
                "status": {"kind": "success", "label": "200 OK"},
                "responseTimeMs": None,
                "usage": None,
                "cost": None,
            }
            commit_user_before_call = not overflow
            if commit_user_before_call:
                self._messages.append({"role": "user", "content": text})
            try:
                started_at = time.monotonic()
                answer, usage = _response_content_and_usage(self._ask_model(payload, **options))
                metadata["responseTimeMs"] = round((time.monotonic() - started_at) * 1000)
                estimated_answer = estimate_text_tokens(answer) if isinstance(answer, str) else 0
                metadata["usage"], metadata["cost"] = request_usage_and_cost(
                    self._settings["model"], usage, estimated_payload, estimated_answer,
                )
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("empty API response")
            except Exception as error:
                metadata["status"] = {"kind": "error", "label": "Ошибка API"}
                self._metadata = metadata
                if overflow:
                    raise OverflowProbeError(provider_error_message(error)) from error
                raise
            if overflow:
                self._messages.append({"role": "user", "content": text})
            self._metadata = metadata
            self._messages.append({"role": "assistant", "content": answer})
            usage_details = metadata["usage"]
            totals = self._metrics["totals"]
            totals["promptTokens"] += usage_details["promptTokens"]
            totals["completionTokens"] += usage_details["completionTokens"]
            totals["totalTokens"] += usage_details["totalTokens"]
            if metadata["cost"]:
                totals["usd"] += metadata["cost"]["usd"]
            record = {
                "messageNumber": len(self._metrics["calls"]) + 1,
                "messageTokens": estimate_text_tokens(text),
                "historyBeforeTokens": estimate_payload_tokens({"messages": self._messages[:-2]}),
                "payloadEstimatedTokens": estimated_payload,
                "promptTokens": usage_details["promptTokens"],
                "completionTokens": usage_details["completionTokens"],
                "totalTokens": usage_details["totalTokens"],
                "source": usage_details["source"],
                "historyAfterTokens": estimate_payload_tokens({"messages": self._messages}),
                "contextLimit": context_limit,
                "contextPercent": context_percent,
                "cost": copy.deepcopy(metadata["cost"]),
                "cumulativePromptTokens": totals["promptTokens"],
                "cumulativeCompletionTokens": totals["completionTokens"],
                "cumulativeTotalTokens": totals["totalTokens"],
                "cumulativeUsd": totals["usd"],
            }
            self._metrics["calls"].append(record)
            self._metrics["lastMessage"] = copy.deepcopy(record)
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
    training_limit = settings["trainingContextLimit"]
    return (
        settings["model"] in MODELS
        and isinstance(settings["systemPrompt"], str)
        and bool(settings["systemPrompt"].strip())
        and settings["format"] in {"text", "json"}
        and (
            max_tokens is None
            or (isinstance(max_tokens, int) and not isinstance(max_tokens, bool)
                and 0 < max_tokens <= MODEL_CAPABILITIES[settings["model"]]["maxOutputTokens"])
        )
        and isinstance(settings["stop"], str)
        and (training_limit is None or (isinstance(training_limit, int) and not isinstance(training_limit, bool) and training_limit > 0))
        and isinstance(settings["sendOnOverflow"], bool)
        and isinstance(settings["contextCompressionEnabled"], bool)
    )


def _valid_nonnegative_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_nonnegative_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _valid_summary_last(last):
    if last is None:
        return True
    if not isinstance(last, dict) or set(last) != {"promptTokens", "completionTokens", "totalTokens", "source", "cost"}:
        return False
    cost = last["cost"]
    return (
        _valid_nonnegative_int(last["promptTokens"])
        and _valid_nonnegative_int(last["completionTokens"])
        and _valid_nonnegative_int(last["totalTokens"])
        and isinstance(last["source"], str)
        and last["source"] in {"actual", "estimated"}
        and (
            cost is None
            or (
                isinstance(cost, dict)
                and set(cost) == {"kind", "usd", "source"}
                and isinstance(cost["kind"], str)
                and cost["kind"] in {"free", "paid"}
                and _valid_nonnegative_number(cost["usd"])
                and isinstance(cost["source"], str)
                and cost["source"] in {"actual", "estimated"}
            )
        )
    )


def _valid_context(context):
    if not isinstance(context, dict) or set(context) != {"summary", "compressedMessageCount", "summaryUsage"}:
        return False
    summary = context["summary"]
    usage = context["summaryUsage"]
    return (
        (summary is None or (isinstance(summary, str) and bool(summary.strip())))
        and _valid_nonnegative_int(context["compressedMessageCount"])
        and ((summary is None) == (context["compressedMessageCount"] == 0))
        and isinstance(usage, dict)
        and set(usage) == {"calls", "promptTokens", "completionTokens", "totalTokens", "usd", "last"}
        and _valid_nonnegative_int(usage["calls"])
        and _valid_nonnegative_int(usage["promptTokens"])
        and _valid_nonnegative_int(usage["completionTokens"])
        and _valid_nonnegative_int(usage["totalTokens"])
        and _valid_nonnegative_number(usage["usd"])
        and _valid_summary_last(usage["last"])
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
        and set(snapshot) == {"id", "name", "messages", "settings", "metadata", "metrics", "context"}
        and _agent_number(snapshot["id"]) is not None
        and isinstance(snapshot["name"], str)
        and bool(snapshot["name"].strip())
        and isinstance(snapshot["messages"], list)
        and all(_valid_message(message) for message in snapshot["messages"])
        and _valid_settings(snapshot["settings"])
        and _valid_metadata(snapshot["metadata"])
        and isinstance(snapshot["metrics"], dict)
        and _valid_context(snapshot["context"])
        and snapshot["context"]["compressedMessageCount"] <= len(snapshot["messages"])
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

    def delete(self, agent_id):
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                return None
            remaining_agents = {
                current_id: current_agent
                for current_id, current_agent in self._agents.items()
                if current_id != agent_id
            }
            state = {
                "version": STATE_VERSION,
                "nextId": self._next_id,
                "agents": [current_agent.snapshot() for current_agent in remaining_agents.values()],
            }
            self._save_state(state)
            self._agents = remaining_agents
            return agent.snapshot()

    def _state(self):
        with self._lock:
            return {
                "version": STATE_VERSION,
                "nextId": self._next_id,
                "agents": [agent.snapshot() for agent in self._agents.values()],
            }

    def _save(self):
        self._save_state(self._state())

    def _save_state(self, state):
        temporary_path = None
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                "w", encoding="utf-8", dir=self._state_path.parent, delete=False,
            ) as temporary_file:
                json.dump(state, temporary_file, ensure_ascii=False)
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
        if isinstance(state, dict) and state.get("version") in {1, 2} and isinstance(state.get("agents"), list):
            for snapshot in state["agents"]:
                if isinstance(snapshot, dict):
                    settings = snapshot.get("settings")
                    if isinstance(settings, dict):
                        settings.setdefault("trainingContextLimit", None)
                        settings.setdefault("sendOnOverflow", False)
                        settings.setdefault("contextCompressionEnabled", True)
                    snapshot.setdefault("metrics", default_metrics())
                    metrics = snapshot.get("metrics")
                    if isinstance(metrics, dict):
                        totals = metrics.get("totals")
                        if not isinstance(totals, dict):
                            metrics["totals"] = default_metrics()["totals"]
                        else:
                            for key, value in default_metrics()["totals"].items():
                                totals.setdefault(key, value)
                    snapshot.setdefault("context", default_context())
            state["version"] = STATE_VERSION
        if (
            not isinstance(state, dict)
            or set(state) != {"version", "nextId", "agents"}
            or state["version"] != STATE_VERSION
            or not isinstance(state["nextId"], int)
            or isinstance(state["nextId"], bool)
            or state["nextId"] < 1
            or not isinstance(state["agents"], list)
            or not all(_valid_agent_snapshot(snapshot) for snapshot in state["agents"])
        ):
            return None
        ids = [snapshot["id"] for snapshot in state["agents"]]
        numbers = [_agent_number(agent_id) for agent_id in ids]
        if len(set(ids)) != len(ids) or (numbers and state["nextId"] <= max(numbers)):
            return None
        return {
            snapshot["id"]: Agent(
                snapshot["id"],
                snapshot["name"],
                snapshot["settings"],
                self._ask_model,
                snapshot["messages"],
                snapshot["metadata"],
                snapshot["metrics"],
                snapshot["context"],
            )
            for snapshot in state["agents"]
        }, state["nextId"]
