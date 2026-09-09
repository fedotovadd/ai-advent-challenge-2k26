# День 7 — сохранение контекста Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Сохранять историю, настройки и метаданные всех агентов в JSON и восстанавливать их между запусками приложения.

**Architecture:** `AgentRegistry` владеет снимком всех агентов и JSON-хранилищем. После каждой изменяющей операции он атомарно записывает версионированный снимок; при создании реестра валидирует и восстанавливает его. HTTP-слой вызывает изменяющие методы реестра и возвращает отдельную ошибку `500`, если сохранение невозможно.

**Tech Stack:** Python 3 standard library (`json`, `pathlib`, `tempfile`, `os`), `unittest`, существующий `ThreadingHTTPServer`.

---

## File structure

Все команды `python3 -m unittest` выполняются из
`deepseek-api`; команды Git и работа с путями документации — из корня
worktree `.../.worktrees/day-07`.

- Modify: `agent.py` — JSON-хранилище, валидация снимка, восстановление `AgentRegistry` и сохранение после мутаций.
- Modify: `web.py` — принимать путь хранилища, обращаться к изменяющим методам реестра и преобразовывать ошибку сохранения в `500`.
- Modify: `main.py` — передавать необязательный путь хранилища при создании сервера в тестах.
- Modify: `tests/test_agent.py` — изолированные тесты сериализации, восстановления, продолжения контекста и невалидного JSON.
- Modify: `tests/test_web.py` — HTTP-тест для понятной ошибки сохранения и сценария перезапуска.
- Modify: `.gitignore` — исключить локальный файл состояния `deepseek-api/data/agents.json`.
- Modify: `README.md`, `deepseek-api/README.md` — отметить ветку и описать ручную проверку перезапуска.

### Task 1: Добавить сохранение состояния реестра

**Files:**
- Modify: `deepseek-api/agent.py:1-186`
- Test: `deepseek-api/tests/test_agent.py:120-165`

- [ ] **Step 1: Написать падающий тест восстановления истории и настроек**

  Добавить тест с `tempfile.TemporaryDirectory`: создать `AgentRegistry(..., state_path)`, изменить настройки `agent-1`, отправить «Меня зовут Маша», затем создать второй реестр с тем же путём. Проверить восстановленные сообщения, настройки и метаданные; отправить «Как меня зовут?» и проверить, что payload содержит обе реплики старого диалога перед новым сообщением.

- [ ] **Step 2: Запустить тест и убедиться, что он падает**

  Run: `python3 -m unittest tests.test_agent.AgentRegistryTests.test_registry_restores_messages_settings_and_context -v`

  Expected: FAIL, поскольку `AgentRegistry` пока не принимает `state_path` и создаёт пустой реестр.

- [ ] **Step 3: Написать минимальную реализацию JSON-хранилища**

  В `agent.py` добавить `PersistenceError` и константу пути по умолчанию:

  ```python
  DEFAULT_STATE_PATH = Path(__file__).with_name("data") / "agents.json"
  ```

  Изменить конструктор на `AgentRegistry(ask_model, state_path=None)` и нормализовать `None` в `DEFAULT_STATE_PATH`. В существующем классе `AgentRegistryTests` добавить `setUp`/`tearDown` с `TemporaryDirectory` и передавать его `agents.json` во *все* конструкторы `AgentRegistry`, чтобы тесты не читали и не меняли рабочий файл приложения. Реализовать `_state()`, `_save()` и `_load()`:

  ```python
  {"version": 1, "nextId": self._next_id, "agents": self.agents()}
  ```

  `_save()` создаёт родительскую директорию, записывает JSON во временный файл в той же директории и завершает запись через `os.replace`. Ошибки записи преобразует в `PersistenceError`.

  `_load()` принимает только снимок с `version == 1`, непустым списком агентов и `nextId`, который больше каждого номера `agent-N`. Для каждого агента проверить `id`, `name`, сообщения с ролями `user`/`assistant` и непустым `content`, все ключи *и значения* `DEFAULT_SETTINGS` (известная модель, непустой system prompt, `text`/`json`, положительный `maxTokens` или `None`, строковый `stop`), а также `metadata` как `None` либо объект со всеми ключами текущих метаданных. Ошибочный, отсутствующий или нечитаемый файл должен вернуть новый `agent-1` без исключения.

  Заменить `threading.Lock` реестра на `threading.RLock`. Каждый публичный метод, который меняет реестр или агента (`create`, `create_many`, `update_settings`, `respond`), удерживает этот lock от изменения до формирования снимка и завершения `_save()`; `agents`, `get` и `_state` используют тот же lock. Так два одновременных HTTP-запроса не смогут записать устаревший снимок поверх более нового. Методы `update_settings(agent_id, settings)` и `respond(agent_id, text, temperature)` вызывают соответствующий метод агента и сохраняют снимок как после успеха, так и после ошибки API, чтобы пользовательское сообщение не потерялось. `create()` и `create_many()` также вызывают `_save()` после изменения списка.

- [ ] **Step 4: Запустить тест и убедиться, что он проходит**

  Run: `python3 -m unittest tests.test_agent.AgentRegistryTests.test_registry_restores_messages_settings_and_context -v`

  Expected: PASS.

- [ ] **Step 5: Написать падающий тест безопасного запуска с некорректным состоянием**

  Создать JSON с отсутствующими настройками, затем создать реестр с этим путём. Проверить единственного стандартного `agent-1` и отсутствие исключения.

- [ ] **Step 6: Запустить тест и убедиться, что он падает**

  Run: `python3 -m unittest tests.test_agent.AgentRegistryTests.test_registry_ignores_invalid_saved_state -v`

  Expected: FAIL до полной валидации снимка.

- [ ] **Step 7: Дополнить валидацию снимка только необходимыми проверками**

  Не добавлять миграции, удаление старых файлов или SQLite. Отклонять снимок целиком при первой структурной ошибке и оставлять файл нетронутым до следующей успешной операции.

- [ ] **Step 8: Запустить оба теста реестра**

  Run: `python3 -m unittest tests.test_agent.AgentRegistryTests -v`

  Expected: PASS.

- [ ] **Step 9: Закоммитить изменения Task 1**

  ```bash
  git add deepseek-api/agent.py deepseek-api/tests/test_agent.py
  git commit -m "feat: persist agent registry state"
  ```

### Task 2: Связать сохранение с HTTP-сервером и сообщить об ошибке записи

**Files:**
- Modify: `deepseek-api/web.py:1-271`
- Modify: `deepseek-api/main.py:1-23`
- Test: `deepseek-api/tests/test_web.py:15-58, 175-448`

- [ ] **Step 1: Написать падающий HTTP-тест сохранения и перезапуска**

  В `DeepSeekWebTests` создавать временный путь в `setUp`, передавать его в `ChatServer` и удалять директорию в `tearDown`. Добавить тест: отправить первое сообщение и настройки через HTTP, остановить сервер, создать второй `ChatServer` с тем же путём и новым тестовым провайдером, отправить второе сообщение. Проверить `200`, восстановленные настройки и полный payload со старой парой сообщений.

- [ ] **Step 2: Запустить тест и убедиться, что он падает**

  Run: `python3 -m unittest tests.test_web.DeepSeekWebTests.test_server_restart_restores_agent_context -v`

  Expected: FAIL, потому что `ChatServer` не принимает путь состояния и HTTP-обработчик вызывает `Agent` напрямую.

- [ ] **Step 3: Сделать минимальные изменения HTTP-слоя**

  Изменить сигнатуры на:

  ```python
  class ChatServer(ThreadingHTTPServer):
      def __init__(self, address, ask_model, state_path=None): ...

  def create_server(host="127.0.0.1", port=8000, ask_model=ask_model, state_path=None): ...
  ```

  Нормализовать `state_path=None` в `ChatServer` перед созданием `AgentRegistry`, чтобы обычный запуск использовал `DEFAULT_STATE_PATH`. В `_handle_message` и `_handle_settings` использовать методы реестра, а не мутацию результата `get()`. Во всех изменяющих HTTP-маршрутах — `_handle_message`, `_handle_settings`, `POST /api/agents` и `_handle_bulk_create` — перехватить `PersistenceError` и вернуть `500` с JSON `{"error": "Не удалось сохранить состояние агента."}`. Этот перехват должен стоять раньше общего `Exception`; существующие коды `400`, `502` и `503` не менять.

- [ ] **Step 4: Запустить тест и убедиться, что он проходит**

  Run: `python3 -m unittest tests.test_web.DeepSeekWebTests.test_server_restart_restores_agent_context -v`

  Expected: PASS.

- [ ] **Step 5: Написать падающий тест ошибки записи**

  Замокировать метод записи JSON-хранилища так, чтобы он выбрасывал `PersistenceError`; отправить валидное сообщение и создание агента через HTTP и проверить в обоих случаях `500` и точное безопасное сообщение без внутренней причины.

- [ ] **Step 6: Запустить тест и убедиться, что он падает**

  Run: `python3 -m unittest tests.test_web.DeepSeekWebTests.test_storage_error_returns_safe_500 -v`

  Expected: FAIL, пока исключение попадает в общий путь `502`.

- [ ] **Step 7: Добавить узкий перехват `PersistenceError`**

  Убедиться, что ошибка сохранения не выдаёт детали файловой системы и не маскируется как ошибка API. Оставить состояние агента в памяти, но не возвращать успешный статус, так как переживание перезапуска не гарантировано.

- [ ] **Step 8: Запустить веб-тесты**

  Run: `python3 -m unittest tests.test_web -v`

  Expected: PASS.

- [ ] **Step 9: Закоммитить изменения Task 2**

  ```bash
  git add deepseek-api/web.py deepseek-api/main.py deepseek-api/tests/test_web.py
  git commit -m "feat: restore agent context on server restart"
  ```

### Task 3: Документировать хранение и выполнить итоговую проверку

**Files:**
- Modify: `.gitignore:1-6`
- Modify: `README.md:3-9`
- Modify: `deepseek-api/README.md:1-60, 194-220`

- [ ] **Step 1: Добавить локальное состояние в `.gitignore`**

  Добавить `deepseek-api/data/agents.json`, чтобы история и метаданные пользователя не попадали в Git.

- [ ] **Step 2: Дополнить README**

  В корневом README добавить ссылку на День 7 и ветку `day-07`. В README приложения изменить фразу о хранении «пока работает приложение» и добавить раздел «День 7 — сохранение контекста»: что хранится JSON-снимок всех агентов и где расположен файл, что API-ключи туда не записываются, а также ручные шаги: отправить сообщение, остановить сервер, запустить `python3 main.py`, отправить уточняющий вопрос и убедиться, что ответ учитывает первую реплику.

- [ ] **Step 3: Проверить, что рабочее состояние не отслеживается Git**

  Run: `git check-ignore -v deepseek-api/data/agents.json`

  Expected: `.gitignore` указывает правило для файла.

- [ ] **Step 4: Запустить полный набор тестов**

  Run: `python3 -m unittest discover -s tests -v`

  Expected: PASS, 0 failures and 0 errors.

- [ ] **Step 5: Проверить итоговый diff**

  Run: `git diff day-06...HEAD --check && git status --short --branch`

  Expected: отсутствие ошибок пробелов; в статусе только ожидаемые изменения документации до коммита.

- [ ] **Step 6: Закоммитить документацию**

  ```bash
  git add .gitignore README.md deepseek-api/README.md
  git commit -m "docs: explain persistent agent context"
  ```
