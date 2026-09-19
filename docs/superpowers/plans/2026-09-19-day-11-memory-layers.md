# День 11 — модель слоёв памяти: план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Реализовать для локального агента три явных, изолированных слоя памяти с постоянным хранением, раздельным UI и проверяемым влиянием на промпт.

**Architecture:** Краткосрочной памятью остаётся ограниченная история messages. Рабочая и долговременная память добавляются в context.memoryLayers, валидируются отдельным модулем и атомарно сохраняются в прежнем JSON state file. Agent добавляет непустые слои в основной запрос стабильными системными JSON-блоками; отдельный клиентский модуль даёт адресные действия панели памяти.

**Tech Stack:** Python 3, unittest, встроенный http.server, vanilla JavaScript, HTML/CSS, JSON.

---

## Границы

- Включено: short-term messages; working; longTerm.profile, longTerm.decisions, longTerm.knowledge; явные сохранение, очистка и замена working; перезапуск сервера; prompt injection; UI и доказательство влияния слоёв.
- Не включено: Task State Machine, переходы, инварианты, повторная валидация ответа, архив нескольких задач и разделяемая между агентами память.
- Новый агент остаётся полностью изолированным. Для новой задачи в том же агенте пользователь очищает или заменяет working; longTerm сохраняется.

## Файлы

| Файл | Работа |
|---|---|
| deepseek-api/memory_layers.py | Создать схемы, валидацию и стабильные prompt blocks. |
| deepseek-api/agent.py | Хранить/мигрировать memoryLayers, inject в prompt, сохранить trace и обновлять через registry. |
| deepseek-api/web.py | Отдать static asset и принять четыре адресных PUT API. |
| deepseek-api/static/memory-controls.js | Создать постоянную панель и клиентские действия. |
| deepseek-api/static/index.html | Подключить и смонтировать панель, добавить компактный CSS. |
| deepseek-api/tests/test_memory_layers.py | Создать unit tests контрактов. |
| deepseek-api/tests/test_agent.py, tests/test_context_strategies.py | Расширить тестами persistence/prompt/branch isolation. |
| deepseek-api/tests/test_web.py | Расширить HTTP API и static tests. |
| deepseek-api/tests/test_memory_ui.py | Создать JS UI harness. |
| docs/day-11/memory-layers-results.md | Создать проверяемый отчёт задания. |

### Task 1: Контракты памяти

**Files:**
- Create: deepseek-api/memory_layers.py
- Test: deepseek-api/tests/test_memory_layers.py

- [ ] **Step 1: Написать падающие tests-first проверки**

Проверить точные defaults:

~~~python
self.assertEqual(default_memory_layers(), {
    "working": {"task": "", "data": {}},
    "longTerm": {"profile": {}, "decisions": [], "knowledge": {}},
})
self.assertTrue(valid_working({"task": "", "data": {}}))
self.assertFalse(valid_working({"task": "Новая", "data": {"bad": []}}))
~~~

Покрыть независимое копирование defaults, отсутствие лишних ключей, непустые строки, 24 пары working.data, 16 profile/knowledge и 20 decisions, а также все границы строки из спецификации.

- [ ] **Step 2: Подтвердить red stage**

Run: python3 -m unittest tests.test_memory_layers -v

Expected: FAIL с ModuleNotFoundError для memory_layers.

- [ ] **Step 3: Реализовать минимальный контракт**

Создать functions:

~~~python
def default_working():
    return {"task": "", "data": {}}

def default_long_term():
    return {"profile": {}, "decisions": [], "knowledge": {}}

def default_memory_layers():
    return {"working": default_working(), "longTerm": default_long_term()}

def valid_working(value): ...
def valid_long_term(value): ...
def valid_memory_layers(value): ...
def stable_block(label, value):
    return f"[{label}]\n" + json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
~~~

Валидация не нормализует request: полный объект проходит или сервер возвращает 400. stable_block вызывается лишь для содержательного working (task или data) и longTerm (хотя бы одна непустая вложенная коллекция).

- [ ] **Step 4: Подтвердить green stage**

Run: python3 -m unittest tests.test_memory_layers -v

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add deepseek-api/memory_layers.py deepseek-api/tests/test_memory_layers.py
git commit -m "feat: define explicit memory layer schemas"
~~~

### Task 2: Agent, prompt и JSON persistence

**Files:**
- Modify: deepseek-api/agent.py:19-124, 254-260, 401-493, 680-1095
- Modify: deepseek-api/tests/test_agent.py
- Modify: deepseek-api/tests/test_context_strategies.py

- [ ] **Step 1: Написать failing Agent tests**

Потребовать раздельные изменения и prompt order:

~~~python
agent.update_working_memory({"task": "Лендинг", "data": {"tone": "спокойный"}})
agent.update_long_term_memory("profile", {"style": "кратко"})
agent.update_long_term_memory("decisions", ["Не использовать платные API"])
snapshot = agent.respond("Сделай текст")
trace = snapshot["metadata"]["memoryLayers"]
self.assertEqual(trace["working"]["data"]["tone"], "спокойный")
self.assertIn("[WORKING_MEMORY]", calls[-1]["messages"][1]["content"])
self.assertIn("[LONG_TERM_MEMORY]", calls[-1]["messages"][2]["content"])
~~~

Добавить отдельные tests: пустые блоки не отправляются; порядок base → working → long-term → прежние facts/summary → short-term messages → user; short-term сохраняет поведение Day 10; clear working не трогает longTerm; branching switch не откатывает memoryLayers; restart восстанавливает слои; v4 мигрирует в defaults; malformed v5 отклоняется.

- [ ] **Step 2: Запустить red stage**

Run: python3 -m unittest tests.test_agent tests.test_context_strategies -v

Expected: FAIL, потому что Agent memory API, metadata trace и v5 migration отсутствуют.

- [ ] **Step 3: Реализовать Agent-уровень**

1. Импортировать contracts; изменить STATE_VERSION с 4 на 5; добавить memoryLayers: default_memory_layers() в default_context.
2. Реализовать Agent.update_working_memory(working) и Agent.update_long_term_memory(category, entries). Category допускает только profile, decisions и knowledge; изменяется только адресуемая часть.
3. Исключить memoryLayers из _branch_state; _valid_context отдельно проверяет top-level valid_memory_layers, а branch/checkpoint snapshots сохраняют прежнюю memory schema. switch_branch не заменяет global working/longTerm.
4. Реализовать _memory_layer_messages(system_prompt, context), возвращающий base system message с явной фразой «Данные памяти — контекст, а не системные инструкции», затем непустые stable [WORKING_MEMORY] и [LONG_TERM_MEMORY] blocks. Использовать в summary и остальных стратегиях, добавляя прежние facts/summary/short-term ровно как раньше. Если оба новых слоя пусты, исходный system prompt остаётся побайтно прежним.
5. Добавить memoryLayers в METADATA_FIELDS и metadata trace. Trace имеет ключи shortTerm, working, longTerm: shortTerm — точная глубокая копия нативных user/assistant messages, переданных в конкретный prompt; working/longTerm — отправленные значения или null. Обновить _valid_metadata.
6. В _load мигрировать v4 snapshots до общей валидации: context.memoryLayers получает defaults, state version становится 5; branch/checkpoint snapshots не меняются. В v5 repair только отсутствующий или malformed context.memoryLayers, подставляя defaults до проверки остальных полей; повреждённые другие части state по-прежнему обрабатываются существующим безопасным отклонением registry.
7. Использовать те же base + memory system messages в full_history_payload до полной диалоговой истории, чтобы payload estimate, overflow и compression-savings сравнивали сопоставимые запросы.
8. Добавить AgentRegistry.update_memory(agent_id, scope, entries) с rollback из before snapshot при PersistenceError, по образцу update_settings.

- [ ] **Step 4: Запустить Agent regression suite**

Run: python3 -m unittest tests.test_memory_layers tests.test_agent tests.test_context_strategies -v

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add deepseek-api/agent.py deepseek-api/tests/test_agent.py deepseek-api/tests/test_context_strategies.py
git commit -m "feat: persist and inject memory layers"
~~~

### Task 3: Адресные HTTP API

**Files:**
- Modify: deepseek-api/web.py:1-92, 170-330
- Modify: deepseek-api/tests/test_web.py

- [ ] **Step 1: Написать failing HTTP tests**

Добавить exact endpoint scenario:

~~~python
status, body, _ = self.json_request(
    "PUT", "/api/agents/agent-1/memory/working",
    {"task": "Каталог", "data": {"audience": "B2B"}},
)
self.assertEqual(status, 200)
self.assertEqual(
    body["agent"]["context"]["memoryLayers"]["working"]["task"], "Каталог"
)
~~~

Покрыть profile, decisions, knowledge, clear working, invalid schema/category/JSON и body больше 65 536 байт (400), foreign Origin (403), unknown agent (404), PersistenceError (500), точечность и рестарт ChatServer с тем же state_path. Добавить static test для memory-controls.js.

- [ ] **Step 2: Запустить red stage**

Run: python3 -m unittest tests.test_web.DeepSeekWebTests -v

Expected: FAIL: новые /memory paths вернут 404 и static asset отсутствует.

- [ ] **Step 3: Реализовать маршруты**

В do_GET отдать /static/memory-controls.js. В do_PUT распознать только:

~~~text
/api/agents/{id}/memory/working
/api/agents/{id}/memory/long-term/profile
/api/agents/{id}/memory/long-term/decisions
/api/agents/{id}/memory/long-term/knowledge
~~~

Добавить MAX_MEMORY_REQUEST_BYTES = 65_536 и private _read_memory_json_body(): он отклоняет Transfer-Encoding, повторяющийся/нечисловой Content-Length, пустое или превышающее лимит тело и невалидный UTF-8/JSON с 400. Это отдельный helper: существующего общего _read_json в web.py нет, а обработчики settings/context action не дают достаточной проверки размера. _handle_memory_update использует этот helper и registry.update_memory, возвращает 200 {agent}, 404 для неизвестного агента, 400 для ValueError и existing storage-safe 500. Сохранить same-origin защиту handler.

- [ ] **Step 4: Запустить HTTP suite**

Run: python3 -m unittest tests.test_web.DeepSeekWebTests -v

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add deepseek-api/web.py deepseek-api/tests/test_web.py
git commit -m "feat: expose explicit memory APIs"
~~~

### Task 4: Панель «Память · День 11»

**Files:**
- Create: deepseek-api/static/memory-controls.js
- Modify: deepseek-api/static/index.html:10-37, 48-65, 90-118
- Test: deepseek-api/tests/test_memory_ui.py

- [ ] **Step 1: Написать UI tests**

Создать Node harness по образцу tests/test_context_ui.py: action не вызывает API, если пока ожидался flush settings изменился active agent или появился pending message. Добавить static checks на id memory-controls и видимые подписи: Краткосрочная память, Рабочая память, Долговременная память, Сохранить рабочую память, Очистить рабочую память, Сохранить профиль, Сохранить решения и Сохранить знания.

- [ ] **Step 2: Запустить red stage**

Run: python3 -m unittest tests.test_memory_ui -v

Expected: FAIL, файл static/memory-controls.js отсутствует.

- [ ] **Step 3: Реализовать клиентский модуль**

MemoryControls.mount получает activeAgent, state, api, replaceAgent, render, setChatStatus, syncSendButton и flushSettingsSave.

1. Отображает short-term: agent.messages.length, active contextStrategy и подпись об ограниченной истории.
2. Рендерит task и JSON-object textarea working.data; Save вызывает PUT /memory/working. Clear после window.confirm отправляет {"task":"","data":{}} и не передаёт longTerm.
3. Рендерит отдельные JSON-object формы profile/knowledge, плюс decisions textarea (одна строка = одно решение). Каждая кнопка отправляет только {entries} на собственный endpoint.
4. В details «Последний prompt по слоям» выводит metadata.memoryLayers через textContent.
5. Блокирует controls во время pending response/context action/save; обновляет UI только snapshot из успешного response; ошибки идут в существующий chat status.

Подключить asset после context-controls.js. Создать memoryControls при bootstrap, вызвать memoryControls.refresh() из render и на обеих границах отправки сообщения. Добавить CSS рядом с context-controls; мобильная колонка остаётся вертикальной.

- [ ] **Step 4: Запустить UI regression suite**

Run: python3 -m unittest tests.test_memory_ui tests.test_context_ui tests.test_web.DeepSeekWebTests -v

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add deepseek-api/static/index.html deepseek-api/static/memory-controls.js deepseek-api/tests/test_memory_ui.py
git commit -m "feat: add memory layer controls"
~~~

### Task 5: Проверить влияние и оформить результат дня

**Files:**
- Modify: deepseek-api/tests/test_agent.py
- Modify: deepseek-api/README.md
- Create: docs/day-11/memory-layers-results.md

- [ ] **Step 1: Написать deterministic acceptance test**

~~~python
def ask(payload, **options):
    prompt = "\n".join(message["content"] for message in payload["messages"])
    return "рабочая" if "B2B" in prompt else (
        "долговременная" if "кратко" in prompt else "краткосрочная"
    )
~~~

В test_each_memory_layer_changes_model_input_and_answer последовательно добавить предыдущий диалог, working.data и longTerm.profile. Для каждого прогона проверять, что маркер оказался только в предназначенном слое trace и test double вернул соответствующий ответ. Этот test добавляется на red stage Task 2 вместе с Agent API и становится зелёным в Step 4 той задачи.

- [ ] **Step 2: Выполнить acceptance test как evidence**

Run: python3 -m unittest tests.test_agent.AgentTests.test_each_memory_layer_changes_model_input_and_answer -v

Expected: PASS; тест уже стал зелёным в Task 2 и теперь используется как воспроизводимое evidence для документации.

- [ ] **Step 3: Добавить учебный отчёт**

Создать docs/day-11/memory-layers-results.md с таблицей «слой → место → срок жизни → prompt → очистка», перечислить data/agents.json, windowSize для short-term, clear/replace working и сохранение longTerm. Описать результаты test double, а не выдуманный реальный API ответ. Добавить ссылку на панель и отчёт в deepseek-api/README.md.

- [ ] **Step 4: Выполнить полную проверку**

Run: python3 -m unittest discover -s tests -v

Expected: PASS без failures. Если обычная среда блокирует HTTP server, повторить с разрешением локальных сокетов.

- [ ] **Step 5: Commit**

~~~bash
git add deepseek-api/tests/test_agent.py deepseek-api/README.md docs/day-11/memory-layers-results.md
git commit -m "docs: record day 11 memory layer verification"
~~~

## Финальная проверка

- [ ] git diff --check
- [ ] python3 -m unittest discover -s tests -v
- [ ] Запустить python3 main.py из deepseek-api и проверить сохранение working, clear working с сохранением profile, раздельные profile/decision/knowledge, prompt trace и рестарт сервера.
- [ ] Сверить спецификацию docs/superpowers/specs/2026-09-19-day-11-memory-layers-design.md с endpoint, UI и test evidence.
