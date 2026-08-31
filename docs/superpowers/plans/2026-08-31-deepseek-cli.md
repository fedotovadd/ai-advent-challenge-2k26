# DeepSeek CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Создать минимальную Python CLI-программу, которая отправляет введённый пользователем запрос в DeepSeek API и печатает ответ.

**Architecture:** Вся исполняемая логика находится в `main.py`: валидация ключа и запроса, создание совместимого с OpenAI клиента, один вызов Chat Completions API и вывод результата. Зависимость объявлена отдельно, а README содержит полный путь от получения ключа до запуска.

**Tech Stack:** Python 3.9+, OpenAI Python SDK, DeepSeek Chat Completions API

---

## Структура файлов

- `main.py` — CLI, проверка входных данных, вызов API и обработка ошибок.
- `requirements.txt` — Python-зависимость приложения.
- `README.md` — инструкции по получению ключа, установке и запуску.

### Task 1: Объявить зависимость

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Создать проверку, которая пока завершается ошибкой**

Run:

```bash
python3 -c "from pathlib import Path; assert Path('requirements.txt').read_text().strip() == 'openai>=1.0.0'"
```

Expected: `FileNotFoundError`, потому что файл ещё не создан.

- [ ] **Step 2: Добавить зависимость**

```text
openai>=1.0.0
```

- [ ] **Step 3: Повторить проверку**

Run:

```bash
python3 -c "from pathlib import Path; assert Path('requirements.txt').read_text().strip() == 'openai>=1.0.0'"
```

Expected: exit code `0` без вывода.

- [ ] **Step 4: Создать виртуальное окружение и установить зависимость**

Run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Expected: виртуальное окружение создано, пакет `openai` установлен без ошибок.

### Task 2: Реализовать CLI

**Files:**
- Create: `main.py`

- [ ] **Step 1: Зафиксировать ожидаемое поведение до реализации**

Run:

```bash
python3 -m py_compile main.py
```

Expected: ошибка `No such file or directory`, потому что `main.py` ещё не создан.

- [ ] **Step 2: Написать минимальную программу**

```python
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

    prompt = input("Введите запрос: ").strip()
    if not prompt:
        print("Ошибка: запрос не может быть пустым.", file=sys.stderr)
        return 1

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as error:
        print(f"Ошибка при обращении к DeepSeek: {error}", file=sys.stderr)
        return 1

    print("\nОтвет DeepSeek:")
    print(response.choices[0].message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Проверить синтаксис**

Run:

```bash
python3 -m py_compile main.py
```

Expected: exit code `0` без вывода.

- [ ] **Step 4: Проверить сообщение при отсутствующем ключе**

Run:

```bash
env -u DEEPSEEK_API_KEY .venv/bin/python main.py
```

Expected: exit code `1` и `Ошибка: задайте переменную окружения DEEPSEEK_API_KEY.` в stderr.

- [ ] **Step 5: Проверить пустой запрос без сетевого вызова**

Run:

```bash
printf '\n' | DEEPSEEK_API_KEY=test-key .venv/bin/python main.py
```

Expected: exit code `1` и `Ошибка: запрос не может быть пустым.` в stderr.

### Task 3: Добавить инструкцию по запуску

**Files:**
- Create: `README.md`

- [ ] **Step 1: Проверить отсутствие инструкции**

Run:

```bash
test -f README.md
```

Expected: exit code `1`, потому что файл ещё не создан.

- [ ] **Step 2: Написать README**

README должен содержать:

```markdown
# Минимальный клиент DeepSeek

Программа принимает запрос в терминале, отправляет его в DeepSeek API и выводит ответ модели.

## 1. Получите API-ключ

Создайте ключ в [DeepSeek Platform](https://platform.deepseek.com/api_keys). Никому не отправляйте ключ и не записывайте его в исходный код.

## 2. Подготовьте окружение

Требуется Python 3.9 или новее.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Добавьте ключ

В macOS или Linux:

```bash
export DEEPSEEK_API_KEY="ваш_ключ"
```

## 4. Запустите программу

```bash
python3 main.py
```

Введите запрос и нажмите Enter. Ответ DeepSeek появится в консоли.
```

- [ ] **Step 3: Проверить обязательные инструкции**

Run:

```bash
python3 -c "from pathlib import Path; text = Path('README.md').read_text(); assert all(value in text for value in ['DEEPSEEK_API_KEY', 'pip install -r requirements.txt', 'python3 main.py'])"
```

Expected: exit code `0` без вывода.

### Task 4: Итоговая проверка

**Files:**
- Verify: `main.py`
- Verify: `requirements.txt`
- Verify: `README.md`

- [ ] **Step 1: Проверить установленную зависимость**

Run:

```bash
.venv/bin/python -c "from openai import OpenAI; print('openai импортирован')"
```

Expected: `openai импортирован`.

- [ ] **Step 2: Повторить локальные проверки**

Run:

```bash
python3 -m py_compile main.py
env -u DEEPSEEK_API_KEY .venv/bin/python main.py
printf '\n' | DEEPSEEK_API_KEY=test-key .venv/bin/python main.py
```

Expected: компиляция успешна; две проверки завершаются с ожидаемыми пользовательскими ошибками до сетевого вызова.

- [ ] **Step 3: Выполнить реальный вызов после получения ключа**

Run:

```bash
export DEEPSEEK_API_KEY="ваш_ключ"
.venv/bin/python main.py
```

Expected: после ввода запроса программа печатает непустой ответ под заголовком `Ответ DeepSeek:`.

Этот шаг остаётся за пользователем, поскольку API-ключ нельзя передавать исполнителю.
