import copy
import math
import threading
import time


MODEL = "deepseek-v4-flash"
MODELS = ("deepseek-v4-flash", "glm-4.7-flash", "deepseek-v4-pro")
MODEL_PRICING = {
    "deepseek-v4-flash": {"input": 0.22, "output": 0.66},
    "glm-4.7-flash": None,
    "deepseek-v4-pro": {"input": 0.66, "output": 1.98},
}
SYSTEM_PROMPT = (
    "Ты полезный AI-помощник. Отвечай ясно, практично и по-русски. "
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
DEMO_PROFILES = (
    ("Краткий помощник", "deepseek-v4-flash", "Отвечай кратко, ясно и по существу."),
    ("Аналитик", "glm-4.7-flash", "Разбирай вопрос по шагам и объясняй вывод."),
    ("Критик", "deepseek-v4-pro", "Проверяй допущения и отмечай возможные ошибки."),
)


class MissingApiKeyError(ValueError):
    def __init__(self, key_name):
        super().__init__(f"missing {key_name}")
        self.key_name = key_name


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
    def __init__(self, agent_id, name, settings, ask_model):
        self._lock = threading.RLock()
        self._id = agent_id
        self._name = name
        self._settings = copy.deepcopy(settings)
        self._ask_model = ask_model
        self._messages = []
        self._metadata = None

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


class AgentRegistry:
    def __init__(self, ask_model):
        self._lock = threading.Lock()
        self._ask_model = ask_model
        self._agents = {
            "agent-1": Agent("agent-1", "Первый агент", default_settings(), ask_model),
        }
        self._next_id = 2

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
            return agent.snapshot()

    def create_many(self, count):
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_BULK_AGENTS:
            raise ValueError("Количество агентов должно быть целым числом от 1 до 100.")
        with self._lock:
            created = []
            for _ in range(count):
                agent_number = self._next_id
                agent_id = f"agent-{agent_number}"
                profile_name, model, system_prompt = DEMO_PROFILES[(agent_number - 2) % len(DEMO_PROFILES)]
                settings = default_settings()
                settings.update({"model": model, "systemPrompt": system_prompt})
                agent = Agent(agent_id, f"{profile_name} {agent_number}", settings, self._ask_model)
                self._agents[agent_id] = agent
                self._next_id += 1
                created.append(agent.snapshot())
            return created
