# День 6: первый агент — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Превратить веб-чат Дня 5 в однопроцессное приложение, где N независимых объектов `Agent` получают запросы, вызывают LLM и отображаются в одном интерфейсе.

**Architecture:** Добавить модуль `agent.py` с изолированными `Agent` и `AgentRegistry`. HTTP-обработчик в `main.py` станет транспортным слоем: маршрутизация, JSON и HTTP-статусы; построение payload, история и метаданные перейдут в `Agent.respond`. Один `AgentRegistry` создаётся в одном `ChatServer` и хранит N агентов в памяти; все используют переданную общую функцию LLM-клиента.

**Tech Stack:** Python 3.9+, стандартные `unittest`, `http.server`, пакет `openai`, встроенный HTML/CSS/JavaScript.

**Specification:** `deepseek-api/docs/superpowers/specs/2026-09-08-day-06-first-agent-design.md`

---

## Границы изменений

| Файл | Изменение |
| --- | --- |
| `deepseek-api/agent.py` | Новый доменный модуль: константы чата, `Agent`, `AgentRegistry`, валидация настроек и ошибки. |
| `deepseek-api/main.py` | Импорт доменного модуля; сохранение `ask_deepseek` и Дня 3; замена `SessionStore` и сессионных HTTP-маршрутов на реестр и маршруты агентов; обновление `PAGE`. |
| `deepseek-api/tests/test_agent.py` | Новые unit-тесты доменной логики без HTTP и реальной сети. |
| `deepseek-api/tests/test_main.py` | Переименование тестов и ожиданий API с sessions на agents; интеграционные тесты bulk-создания и интерфейса. |
| `deepseek-api/README.md` | Раздел «День 6 — первый агент», запуск и границы решения. |

Не добавлять базу данных, внешний фреймворк, второй сервер, массовую отправку запросов всем агентам или параллельный оркестратор.

### Task 1: Подготовить изолированную ветку и подтвердить исходное состояние

**Files:**
- Create: отдельный worktree для ветки `day-06`, созданной от текущего `day-05`
- Verify: `deepseek-api/tests/test_main.py`

- [ ] **Step 1: Убедиться, что `day-05` содержит утверждённые ТЗ и план, а пользовательские `.idea/` и `.superpowers/` не включены в коммиты.**

Run: `git status --short --branch && git log --oneline -3`

Expected: текущая ветка `day-05` содержит документацию Дня 6; неотслеживаемые служебные папки остаются нетронутыми.

- [ ] **Step 2: Создать изолированный worktree с веткой `day-06` от `day-05` по инструкции `@using-git-worktrees`.**

Run: `git worktree add <проверенный-путь> -b day-06 day-05`

Expected: новый каталог работает на ветке `day-06`, исходная рабочая папка остаётся на `day-05`.

- [ ] **Step 3: Запустить исходные тесты.**

Run: `cd deepseek-api && python3 -m unittest discover -s tests -v`

Expected: все 35 существующих тестов проходят до изменений.

### Task 2: Реализовать изолированную сущность `Agent`

**Files:**
- Create: `deepseek-api/agent.py`
- Create: `deepseek-api/tests/test_agent.py`

- [ ] **Step 1: Написать failing unit-тесты агента с подставным LLM-клиентом.**

Покрыть три сценария:

```python
def test_agent_responds_with_its_config_and_history():
    calls = []
    agent = Agent("agent-1", "Первый агент", default_settings(), lambda payload, **options: calls.append((payload, options)) or "Ответ")

    result = agent.respond("  Привет  ", 0.4)

    assert result["messages"] == [
        {"role": "user", "content": "Привет"},
        {"role": "assistant", "content": "Ответ"},
    ]
    assert calls[0][0]["messages"][0]["role"] == "system"
    assert calls[0][0]["temperature"] == 0.4

def test_agent_does_not_append_assistant_message_when_client_fails():
    ...

def test_agent_keeps_json_options_and_metadata_on_its_own_snapshot():
    ...
```

- [ ] **Step 2: Запустить unit-тесты и убедиться, что они падают из-за отсутствующего модуля.**

Run: `cd deepseek-api && python3 -m unittest discover -s tests -p 'test_agent.py' -v`

Expected: FAIL с ошибкой импорта `agent`.

- [ ] **Step 3: Создать `agent.py` с минимальной доменной логикой.**

Реализовать:

```python
class Agent:
    def __init__(self, agent_id, name, settings, ask_model): ...
    def snapshot(self): ...
    def update_settings(self, settings): ...
    def respond(self, text, temperature): ...
    def record_error(self): ...

def validate_settings(data): ...
def default_settings(): ...
```

`respond` обязан: обрезать и проверять текст/temperature; добавить user message только в свою историю; собрать `messages` из system prompt и своей истории; применить JSON/max tokens/stop; вызвать `ask_model`; измерить время; получить usage и стоимость; добавить assistant message лишь при непустом ответе; вернуть snapshot и metadata. При ошибке клиента агент сам сохраняет error metadata, не добавляет assistant message и повторно выбрасывает исключение, чтобы HTTP-слой выбрал статус ответа. Использовать отдельный lock агента, чтобы история и metadata одного агента не смешались при двух одновременных HTTP-запросах.

Перенести в модуль агента связанные с чатом константы и чистые помощники: доступные модели, тарифы, `SYSTEM_PROMPT`, настройки по умолчанию, JSON-инструкцию, расчёт usage/cost и извлечение content. `main.py` затем импортирует их, поэтому День 3 сохранит текущие значения.

- [ ] **Step 4: Запустить unit-тесты.**

Run: `cd deepseek-api && python3 -m unittest discover -s tests -p 'test_agent.py' -v`

Expected: PASS.

- [ ] **Step 5: Закоммитить доменный слой.**

Run: `git add deepseek-api/agent.py deepseek-api/tests/test_agent.py && git commit -m "feat: add stateful agent entity"`

### Task 3: Добавить `AgentRegistry` и доказать изоляцию N агентов

**Files:**
- Modify: `deepseek-api/agent.py`
- Modify: `deepseek-api/tests/test_agent.py`

- [ ] **Step 1: Написать failing unit-тесты реестра.**

```python
def test_registry_creates_requested_number_of_unique_agents():
    registry = AgentRegistry(fake_model)
    created = registry.create_many(3)
    assert [agent["id"] for agent in created] == ["agent-2", "agent-3", "agent-4"]
    assert len(registry.agents()) == 4  # с начальным agent-1

def test_settings_and_messages_do_not_leak_between_agents():
    ...

def test_registry_rejects_zero_and_too_large_bulk_count():
    ...
```

- [ ] **Step 2: Запустить новые тесты.**

Run: `cd deepseek-api && python3 -m unittest discover -s tests -p 'test_agent.py' -v`

Expected: FAIL, так как `AgentRegistry` ещё не определён.

- [ ] **Step 3: Реализовать потокобезопасный `AgentRegistry`.**

Использовать `threading.Lock` и словарь агентов. Реализовать `agents()`, `get(agent_id)`, `create()` и `create_many(count)`. Реестр создаёт `agent-1` при запуске. `create_many` ограничивает count константой `MAX_BULK_AGENTS = 100`; для каждой новой сущности задаёт уникальные ID и циклически назначает различные имя/model/system prompt из небольшого набора профилей. Создание не вызывает LLM-клиент.

- [ ] **Step 4: Запустить unit-тесты агента и реестра.**

Run: `cd deepseek-api && python3 -m unittest discover -s tests -p 'test_agent.py' -v`

Expected: PASS.

- [ ] **Step 5: Закоммитить реестр.**

Run: `git add deepseek-api/agent.py deepseek-api/tests/test_agent.py && git commit -m "feat: manage independent agents in one registry"`

### Task 4: Перевести HTTP API с сессий на агентов

**Files:**
- Modify: `deepseek-api/main.py`
- Modify: `deepseek-api/tests/test_main.py`

- [ ] **Step 1: Заменить в `test_main.py` сессионные ожидания на агентские и добавить failing интеграционные тесты.**

Переименовать маршруты в существующих проверках:

```python
GET  /api/agents
POST /api/agents
PUT  /api/agents/agent-1/settings
POST /api/agents/agent-1/messages
```

Добавить:

```python
def test_bulk_create_returns_n_independent_agents(self):
    status, body, _ = self.json_request("POST", "/api/agents/bulk", {"count": 3})
    self.assertEqual(status, 201)
    self.assertEqual(len(body["agents"]), 3)
    self.assertEqual(len({agent["id"] for agent in body["agents"]}), 3)
    self.assertEqual(len(self.server.registry.agents()), 4)

def test_bulk_create_rejects_invalid_count(self):
    ...
```

- [ ] **Step 2: Запустить HTTP-тесты.**

Run: `cd deepseek-api && python3 -m unittest tests.test_main -v`

Expected: FAIL: новые `/api/agents` маршруты ещё отсутствуют.

- [ ] **Step 3: Упростить `main.py` до транспортной роли.**

Сделать следующие изменения:

- удалить `SessionStore` и перенесённые чистые чат-помощники из `main.py`;
- импортировать `AgentRegistry`, константы и helpers из `agent.py`;
- в `ChatServer.__init__` создать `self.registry = AgentRegistry(ask_model)`, сохранив `self.ask_model` для Дня 3;
- заменить `/api/sessions` маршрутами `/api/agents`;
- добавить `POST /api/agents/bulk`, принимающий строго `{ "count": int }`;
- в обработчике messages найти агента и вызвать только `agent.respond(text, temperature)`; обработчик не должен собирать payload и добавлять сообщения;
- изменить `ask_deepseek`, чтобы отсутствие ключа выбрасывало доменную `MissingApiKeyError`; обработчик ловит её и возвращает 503, а остальные ошибки — 502. Агент уже сохранил metadata ошибки, поэтому обработчик только возвращает его snapshot;
- оставить same-origin, проверку JSON и маршруты Дня 3 без изменения поведения.

`Agent.snapshot()` должен возвращать `{id, name, messages, settings, metadata}`. Удалить поведение «первое сообщение меняет title»: имя агента является частью его конфигурации, а не производной от первого текста.

- [ ] **Step 4: Запустить HTTP-тесты.**

Run: `cd deepseek-api && python3 -m unittest tests.test_main -v`

Expected: PASS, включая ошибки API, JSON-настройки, историю одного агента и isolation N агентов.

- [ ] **Step 5: Закоммитить API.**

Run: `git add deepseek-api/main.py deepseek-api/tests/test_main.py && git commit -m "feat: route chat requests through agents"`

### Task 5: Обновить общий веб-интерфейс

**Files:**
- Modify: `deepseek-api/main.py` (строка `PAGE`)
- Modify: `deepseek-api/tests/test_main.py`

- [ ] **Step 1: Добавить failing проверки HTML и клиентского JavaScript.**

Проверить, что страница содержит `Агенты`, `Количество агентов`, элементы `agent-count-input` и `create-many-agents`, а JavaScript использует `/api/agents`, `/api/agents/bulk` и `/messages` с активным ID агента. Удалить проверки названий «сессия» и `#new-chat`.

- [ ] **Step 2: Запустить целевые UI-проверки.**

Run: `cd deepseek-api && python3 -m unittest tests.test_main.DeepSeekWebTests.test_page_contains_agent_controls -v`

Expected: FAIL: новые элементы ещё отсутствуют.

- [ ] **Step 3: Адаптировать `PAGE` без изменения общей компоновки.**

В левой панели заменить «Новый чат»/«Чаты» на «Создать агента»/«Агенты»; добавить numeric input `agent-count-input` с безопасным начальным значением и кнопку `create-many-agents`; рядом отрисовывать общее число агентов. Переименовать в JS `sessions`, `activeSession`, `loadSessions`, `upsertSession` и маршруты в агентские аналоги. В списке отображать `agent.name`, model и число сообщений; заголовок активного чата — имя агента. Настройки и история переключаются по `activeAgentId` и отправляют данные только выбранному агенту.

Не добавлять кнопку «Запустить всех», индикатор параллельных задач или автоматические LLM-вызовы при bulk-создании.

- [ ] **Step 4: Запустить UI и API-тесты.**

Run: `cd deepseek-api && python3 -m unittest tests.test_main -v`

Expected: PASS.

- [ ] **Step 5: Запустить сервер вручную и выполнить короткую проверку в браузере.**

Run: `cd deepseek-api && python3 main.py`

Expected: на `http://127.0.0.1:8000` видно счётчик, создание N агентов не вызывает API-ответов, выбор агента меняет правую панель, а запрос выбранному агенту добавляет только его историю.

- [ ] **Step 6: Закоммитить интерфейс.**

Run: `git add deepseek-api/main.py deepseek-api/tests/test_main.py && git commit -m "feat: manage agents in shared chat interface"`

### Task 6: Обновить документацию и выполнить финальную проверку

**Files:**
- Modify: `README.md`
- Modify: `deepseek-api/README.md`

- [ ] **Step 1: Написать документацию Дня 6.**

В корневом README добавить ссылку на День 6 и ветку `day-06`. В `deepseek-api/README.md` описать: что агент — отдельный объект с собственной конфигурацией и историей; что N агентов работают внутри одного процесса; как создать одного и N агентов в интерфейсе; что bulk-создание не отправляет N запросов LLM; ограничения in-memory и отсутствие массового запуска.

- [ ] **Step 2: Запустить полный набор тестов.**

Run: `cd deepseek-api && python3 -m unittest discover -s tests -v`

Expected: PASS: все адаптированные прежние проверки и новые unit-/bulk-проверки проходят.

- [ ] **Step 3: Проверить рабочее дерево и diff.**

Run: `git diff --check && git status --short && git log --oneline day-05..HEAD`

Expected: нет ошибок пробелов; изменены только файлы из плана; история содержит отдельные осмысленные коммиты Дня 6.

- [ ] **Step 4: Закоммитить документацию.**

Run: `git add README.md deepseek-api/README.md && git commit -m "docs: describe day 06 agents"`
