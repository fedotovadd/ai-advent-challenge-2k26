# Response Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить в веб-чат DeepSeek сессионные настройки system prompt, формата ответа, лимита токенов и стоп-последовательности с отображением фактического API-запроса.

**Architecture:** `SessionStore` будет хранить сериализуемые настройки вместе с каждой сессией. Маршрут `PUT /api/sessions/{id}/settings` валидирует и сохраняет их, а обработчик сообщений строит OpenAI-совместимый payload и keyword-аргументы только из заполненных значений. Встроенная страница получает форму настройки справа, а существующий блок метаданных остаётся ниже неё.

**Tech Stack:** Python 3, `http.server`, OpenAI Python SDK, vanilla HTML/CSS/JavaScript, `unittest`.

---

## File structure

- `main.py` — модель сессии, HTTP-маршрут настроек, сборка вызова DeepSeek и встроенный UI.
- `tests/test_main.py` — проверки API контрактов, формирования вызова модели и наличия UI-контролов.
- `README.md` — инструкция по использованию новых элементов панели.

### Task 1: Сессионные настройки и HTTP-контракт

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py`

- [ ] **Step 1: Update existing response-shape expectations and write failing settings tests**

```python
def test_settings_are_stored_per_session(self):
    status, body, _ = self.json_request(
        "PUT", "/api/sessions/session-1/settings",
        {"systemPrompt": "Новый prompt", "format": "json", "maxTokens": 300, "stop": "<END>"},
    )
    self.assertEqual(status, 200)
    self.assertEqual(body["session"]["settings"]["format"], "json")
```

Update existing assertions that compare whole session objects (`test_initial_session_is_available`, session creation and message-response tests) so each expected session includes `DEFAULT_SETTINGS`. Add cases for an invalid `maxTokens`, whitespace-only `systemPrompt`, and an unknown session; each invalid request must preserve previous settings.

- [ ] **Step 2: Run the focused tests and confirm they fail because the route and settings do not exist**

Run: `python3 -m unittest tests.test_main.DeepSeekWebTests.test_settings_are_stored_per_session -v`

Expected: FAIL with 404 or missing `settings`.

- [ ] **Step 3: Add default settings, deep-copy support and `SessionStore.update_settings`**

Implement a `DEFAULT_SETTINGS` value and add a `settings` field to every newly created session. `update_settings` must update only a known session and return a copy.

- [ ] **Step 4: Add `PUT /api/sessions/{id}/settings` and request validation**

Accept exactly `systemPrompt`, `format`, `maxTokens`, and `stop`; strip strings; permit `text`/`json`, a positive integer or `null`, and reject empty system prompts. Return 400 without updating on invalid input.

- [ ] **Step 5: Run focused and complete tests**

Run: `python3 -m unittest discover -s tests -v`

Expected: PASS, including existing behavior.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: save response settings per session"
```

### Task 2: Сборка параметров DeepSeek и метаданных

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py`

- [ ] **Step 1: Change the injected model fake and write failing tests for a controlled JSON request**

Replace the single-argument fake with `ask_model(payload, **kwargs)` and record `{"payload": payload, "kwargs": kwargs}` in `self.calls`. Update existing assertions that use `self.payloads` to inspect `self.calls[-1]["payload"]` instead.

```python
def test_json_settings_are_sent_to_the_model(self):
    settings = {"systemPrompt": "Верни данные", "format": "json", "maxTokens": 300, "stop": "<END>"}
    self.json_request("PUT", "/api/sessions/session-1/settings", settings)
    self.json_request("POST", "/api/sessions/session-1/messages", {"text": "Привет"})
    self.assertEqual(self.calls[-1]["kwargs"]["response_format"], {"type": "json_object"})
```

- [ ] **Step 2: Run the focused test and confirm it fails because settings do not influence the model call**

Run: `python3 -m unittest tests.test_main.DeepSeekWebTests.test_json_settings_are_sent_to_the_model -v`

Expected: FAIL because the recorded model call has no response format.

- [ ] **Step 3: Extract a small payload/call-options builder**

Build the actual system message from the saved prompt; append a JSON-only instruction only in JSON mode. Add `response_format`, `max_tokens`, and `stop` only when configured. Adapt the injected model callable so tests record the payload and keyword arguments while production passes them to the OpenAI SDK.

- [ ] **Step 4: Include final payload and system prompt in metadata**

The metadata’s `payload` must show the exact model, messages and optional parameters actually submitted to DeepSeek.

- [ ] **Step 5: Run focused and complete tests**

Run: `python3 -m unittest discover -s tests -v`

Expected: PASS; text mode must still omit optional parameters.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: apply response controls to DeepSeek requests"
```

### Task 3: Правая панель настроек

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py`

- [ ] **Step 1: Write a failing HTML contract test**

```python
def test_page_contains_response_settings_and_metadata(self):
    status, body, _ = self.request("GET", "/")
    self.assertIn("Настройки ответа", body)
    self.assertIn("system-prompt-input", body)
    self.assertIn("max-tokens-input", body)
    self.assertIn("Метаданные", body)
```

- [ ] **Step 2: Run the focused test and confirm it fails because the controls are missing**

Run: `python3 -m unittest tests.test_main.DeepSeekWebTests.test_page_contains_response_settings_and_metadata -v`

Expected: FAIL with the absent control identifier.

- [ ] **Step 3: Add controls and responsive styles to the right panel**

Place the settings form before the unchanged metadata cards. Use labels and clear placeholders; retain the existing mobile single-column breakpoint.

- [ ] **Step 4: Connect the form to loaded session settings and the PUT route**

On session selection populate controls. On save, show an inline success/error state and update the in-memory session returned by the server. Do not discard current metadata.

- [ ] **Step 5: Run the complete tests**

Run: `python3 -m unittest discover -s tests -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add response settings panel"
```

### Task 4: Документация и финальная проверка

**Files:**
- Modify: `README.md`
- Modify: `day-01-deepseek-api/README.md`

- [ ] **Step 1: Document response controls**

Describe how each setting applies to the current session, how to compare unrestricted and controlled requests, and the JSON-mode instruction requirement.

- [ ] **Step 2: Run the full test suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 3: Launch the local app and inspect the browser**

Run: `python3 main.py`

Expected: the right panel displays settings above existing metadata; save updates the active session and a controlled request reveals the optional fields in metadata.

- [ ] **Step 4: Commit from repository root**

```bash
cd ..
git add README.md day-01-deepseek-api/README.md
git commit -m "docs: explain response controls"
```
