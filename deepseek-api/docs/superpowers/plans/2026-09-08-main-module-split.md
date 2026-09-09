# Логическое разбиение main.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Разделить `main.py` на модули с одной ответственностью, сохранив запуск, HTTP API и совместимые импорты.

**Architecture:** `agent.py` остаётся доменным слоем и единственным источником общих настроек. Вызов API, эксперимент Дня 3, HTML-страница и HTTP-транспорт переходят соответственно в `providers.py`, `day_three.py`, `page.py` и `web.py`; `main.py` только собирает приложение и реэкспортирует прежний публичный API.

**Tech Stack:** Python, стандартные `unittest` и `http.server`, пакет `openai`.

---

### Task 1: Зафиксировать совместимый публичный API

**Files:**
- Modify: `deepseek-api/tests/test_main.py`
- Modify: `deepseek-api/agent.py`
- Create: `deepseek-api/errors.py`, `deepseek-api/providers.py`
- Modify: `deepseek-api/main.py`

- [ ] Добавить тест, который импортирует `providers` и `errors`, проверяет реэкспорты `main.ask_deepseek` и `main.MissingApiKeyError`, а также прежние константы чата.
- [ ] Запустить этот тест и убедиться, что он падает из-за отсутствующих модулей.
- [ ] Вынести `MissingApiKeyError` в `errors.py`, заменить его определение импортом в `agent.py`, а вызов OpenAI API — в `providers.py`; сделать `main.py` совместимым фасадом.
- [ ] Перенести patch в тесте c `main.OpenAI` на `providers.OpenAI` и повторно запустить тест.

### Task 2: Вынести эксперимент и HTML-интерфейс

**Files:**
- Modify: `deepseek-api/tests/test_main.py`
- Create: `deepseek-api/day_three.py`, `deepseek-api/page.py`
- Modify: `deepseek-api/main.py`

- [ ] Добавить тесты импорта `day_three` и `page`, проверяющие, что фасад `main.py` сохраняет `DAY_THREE_TASKS`, `PAGE` и все системные промпты стратегий (`DIRECT_SYSTEM`, `STEPWISE_SYSTEM`, `PROMPT_ENGINEER_SYSTEM`, `PROMPT_EXECUTOR_SYSTEM`, `ANALYST_SYSTEM`, `ENGINEER_SYSTEM`, `CRITIC_SYSTEM`, `MODERATOR_SYSTEM`).
- [ ] Запустить тесты и зафиксировать ожидаемую ошибку отсутствующих модулей.
- [ ] Перенести константы задач, промпты и функции эксперимента в `day_three.py`, импортируя общие настройки из `agent.py`.
- [ ] Перенести строку `PAGE` в `page.py`, затем заменить код в `main.py` импортами и реэкспортами.
- [ ] Запустить unit-тесты Дня 3 и проверки HTML.

### Task 3: Вынести HTTP-транспорт и собрать точку входа

**Files:**
- Modify: `deepseek-api/tests/test_main.py`
- Create: `deepseek-api/web.py`
- Modify: `deepseek-api/main.py`

- [ ] Добавить тест, подтверждающий, что `create_server()` из `main.py` создаёт `web.ChatServer` с прежними `registry` и `ask_model`.
- [ ] Запустить тест и убедиться, что он падает до переноса класса сервера.
- [ ] Перенести `ChatRequestHandler` и `ChatServer` в `web.py`, передавая HTML, провайдер и функции эксперимента через явные импорты.
- [ ] Передавать `ask_model` только через конструктор `ChatServer`; обработчики используют `self.server.ask_model` и не захватывают `ask_deepseek`. Оставить в `main.py` совместимые реэкспорты, `create_server()` и `main()`.
- [ ] Запустить все тесты, включая HTTP-маршруты, NDJSON и строгую валидацию тел запросов.

### Task 4: Проверить результат

**Files:**
- Verify: `deepseek-api/agent.py`, `deepseek-api/errors.py`, `deepseek-api/providers.py`, `deepseek-api/day_three.py`, `deepseek-api/page.py`, `deepseek-api/web.py`, `deepseek-api/main.py`

- [ ] Запустить `python3 -m py_compile` для всех модулей.
- [ ] Запустить `python3 -m unittest discover -s tests -v`.
- [ ] Запустить `git diff --check` и сверить, что `.idea/` не затронута.
