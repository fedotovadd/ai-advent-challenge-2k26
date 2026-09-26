import os

from openai import OpenAI

from errors import MissingApiKeyError


def _provider_settings(model):
    if model == "glm-4.7-flash":
        return "ZAI_API_KEY", "https://api.z.ai/api/paas/v4"
    return "DEEPSEEK_API_KEY", "https://api.deepseek.com"


def ask_model(payload, **options):
    key_name, base_url = _provider_settings(payload["model"])
    api_key = os.getenv(key_name)
    if not api_key:
        raise MissingApiKeyError(key_name)
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=payload["model"],
        messages=payload["messages"],
        temperature=payload["temperature"],
        **options,
    )
    if not response.choices:
        raise ValueError("empty API response")
    message = response.choices[0].message
    tool_calls = [
        {"id": tool.id, "type": tool.type,
         "function": {"name": tool.function.name, "arguments": tool.function.arguments}}
        for tool in (getattr(message, "tool_calls", None) or [])
    ]
    if not message.content and not tool_calls:
        raise ValueError("empty API response")
    assistant = {"role": "assistant", "content": message.content}
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    return {"content": message.content, "usage": response.usage,
            "tool_calls": tool_calls, "message": assistant}
