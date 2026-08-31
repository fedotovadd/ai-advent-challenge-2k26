import os
import sys

from openai import OpenAI


def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print(
            "Ошибка: задайте переменную окружения DEEPSEEK_API_KEY.",
            file=sys.stderr,
        )
        return 1

    try:
        prompt = input("Введите запрос: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nОшибка: ввод отменён.", file=sys.stderr)
        return 1

    if not prompt:
        print("Ошибка: запрос не может быть пустым.", file=sys.stderr)
        return 1

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        print("Ожидаем ответ DeepSeek...", flush=True)
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("DeepSeek вернул пустой ответ.")
        answer = response.choices[0].message.content
    except Exception as error:
        print(f"Ошибка при обращении к DeepSeek: {error}", file=sys.stderr)
        return 1

    print("\nОтвет DeepSeek:")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
