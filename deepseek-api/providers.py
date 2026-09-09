import os

from openai import OpenAI

from errors import MissingApiKeyError


def _provider_settings(model):
    if model == "glm-4.7-flash":
        return "ZAI_API_KEY", "https://api.z.ai/api/paas/v4"
    return "DEEPSEEK_API_KEY", "https://api.deepseek.com"


def ask_deepseek(payload, **options):
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
    if not response.choices or not response.choices[0].message.content:
        raise ValueError("empty API response")
    return {"content": response.choices[0].message.content, "usage": response.usage}
