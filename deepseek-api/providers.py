import os

from openai import APIConnectionError, APITimeoutError, NotFoundError, OpenAI

from errors import MissingApiKeyError


LOCAL_MODEL = "qwen2.5:1.5b"
LOCAL_BASE_URL = "http://127.0.0.1:11434/v1/"


class LocalModelError(RuntimeError):
    def __init__(self, kind):
        self.kind = kind
        super().__init__(kind)


def _provider_settings(model):
    if model == "glm-4.7-flash":
        return "ZAI_API_KEY", "https://api.z.ai/api/paas/v4"
    return "DEEPSEEK_API_KEY", "https://api.deepseek.com"


def ask_model(payload, **options):
    local = payload["model"] == LOCAL_MODEL
    if local:
        api_key, base_url = "ollama", LOCAL_BASE_URL
        client_options = {"timeout": 120, "max_retries": 0}
    else:
        key_name, base_url = _provider_settings(payload["model"])
        api_key = os.getenv(key_name)
        if not api_key:
            raise MissingApiKeyError(key_name)
        client_options = {}
    if 'max_retries' in options:
        requested_retries = options.pop('max_retries')
        if not local:
            client_options['max_retries'] = requested_retries
    client = OpenAI(api_key=api_key, base_url=base_url, **client_options)
    try:
        response = client.chat.completions.create(
            model=payload["model"],
            messages=payload["messages"],
            temperature=payload["temperature"],
            **options,
        )
    except APITimeoutError as error:
        if local:
            raise LocalModelError("timeout") from error
        raise
    except APIConnectionError as error:
        if local:
            raise LocalModelError("unavailable") from error
        raise
    except NotFoundError as error:
        if local:
            raise LocalModelError("missing_model") from error
        raise
    if not response.choices:
        if local:
            raise LocalModelError("empty")
        raise ValueError("empty API response")
    message = response.choices[0].message
    tool_calls = [
        {"id": tool.id, "type": tool.type,
         "function": {"name": tool.function.name, "arguments": tool.function.arguments}}
        for tool in (getattr(message, "tool_calls", None) or [])
    ]
    if not message.content and not tool_calls:
        if local:
            raise LocalModelError("empty")
        raise ValueError("empty API response")
    assistant = {"role": "assistant", "content": message.content}
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    return {"content": message.content, "usage": response.usage,
            "tool_calls": tool_calls, "message": assistant}
