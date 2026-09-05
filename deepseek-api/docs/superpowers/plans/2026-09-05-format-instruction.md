# Format Instruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить необязательную инструкцию свободного формата, убрать автоматическую JSON-инструкцию и сохранять настройки сессии автоматически.

**Architecture:** Настройки сессии получают строку `formatInstruction`. Сервер применяет её только к текстовому формату и передаёт JSON через `response_format` без изменения system prompt. Клиент скрывает поле инструкции в JSON-режиме, отправляет изменения с задержкой 500 мс и принудительно ждёт сохранения перед отправкой сообщения.

**Tech Stack:** Python 3, `http.server`, vanilla JavaScript, OpenAI Python SDK, `unittest`.

---

## File structure

- `main.py` — настройки сессий, создание API-запроса и встроенная правая панель с автосохранением.
- `tests/test_main.py` — проверки контракта настроек, payload и HTML-интерфейса.
- `README.md` — описание инструкции свободного формата и автосохранения.

### Task 1: Настройки инструкции и payload DeepSeek

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py`

- [ ] **Step 1: Write failing tests for the new settings field**

```python
def test_empty_format_instruction_is_saved(self):
    settings = {"systemPrompt": "База", "format": "text", "formatInstruction": "", "maxTokens": None, "stop": ""}
    status, body, _ = self.json_request("PUT", "/api/sessions/session-1/settings", settings)
    self.assertEqual(status, 200)
    self.assertEqual(body["session"]["settings"]["formatInstruction"], "")
```

Add a 400 case for non-string `formatInstruction`, ensuring stored settings are unchanged.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `python3 -m unittest tests.test_main.DeepSeekWebTests.test_empty_format_instruction_is_saved -v`

Expected: FAIL because the settings contract lacks `formatInstruction`.

- [ ] **Step 3: Add `formatInstruction` to defaults and validation**

Extend `DEFAULT_SETTINGS` and `_validate_settings`; allow an empty string, reject non-strings, and strip nonempty values before storage. Update complete-session expectations in existing tests.

- [ ] **Step 4: Write failing payload tests for both formats**

```python
def test_text_format_appends_instruction_to_system_prompt(self):
    self._save_settings(format="text", format_instruction="Три пункта.")
    self.json_request("POST", "/api/sessions/session-1/messages", {"text": "Привет"})
    self.assertEqual(self.calls[-1]["payload"]["messages"][0]["content"], "База\n\nТри пункта.")

def test_json_format_uses_base_prompt_without_auto_instruction(self):
    self._save_settings(format="json", format_instruction="Не использовать")
    self.json_request("POST", "/api/sessions/session-1/messages", {"text": "Привет"})
    self.assertEqual(self.calls[-1]["payload"]["messages"][0]["content"], "База")
    self.assertEqual(self.calls[-1]["kwargs"]["response_format"], {"type": "json_object"})
```

- [ ] **Step 5: Run focused tests and confirm failure**

Run: `python3 -m unittest tests.test_main.DeepSeekWebTests.test_text_format_appends_instruction_to_system_prompt tests.test_main.DeepSeekWebTests.test_json_format_uses_base_prompt_without_auto_instruction -v`

Expected: FAIL because current code appends the old JSON instruction and ignores the new text instruction.

- [ ] **Step 6: Build the actual system prompt by selected format**

For `text`, append a nonempty saved instruction with a blank line; for `json`, retain only the base prompt and add `response_format`. Delete `JSON_OUTPUT_INSTRUCTION`.

- [ ] **Step 7: Run the complete suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add free-format instruction"
```

### Task 2: Условная панель и автосохранение

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py`

- [ ] **Step 1: Write a failing HTML contract test**

```python
def test_page_contains_autosaved_format_instruction(self):
    _, body, _ = self.request("GET", "/")
    self.assertIn("format-instruction-input", body)
    self.assertIn("scheduleSettingsSave", body)
    self.assertNotIn("save-settings", body)
```

- [ ] **Step 2: Run focused test and confirm failure**

Run: `python3 -m unittest tests.test_main.DeepSeekWebTests.test_page_contains_autosaved_format_instruction -v`

Expected: FAIL because the page still has a save button and lacks the new field/autosave handler.

- [ ] **Step 3: Replace the save button with an autosave status**

Add the instruction textarea directly after the format selector; hide it when format is JSON. Remove the save button. Add listeners that save format immediately and debounce text/number inputs by 500 ms.

- [ ] **Step 4: Make autosave session-scoped and composer submit wait for dirty settings**

Extract one `saveSettings(sessionId)` promise function. Capture the active session ID when scheduling the debounce; ignore a stale timer if the active session has changed. Before switching sessions, flush the pending save for the old session. If there are unsaved form changes on composer submit, cancel the debounce and await this function; send the message only on success. Populate fields from normalized server response after every successful save.

- [ ] **Step 5: Run complete tests**

Run: `python3 -m unittest discover -s tests -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: autosave response settings"
```

### Task 3: Документация и визуальная проверка

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the settings documentation**

Document the optional free-format instruction, automatic saving, JSON’s direct API parameter, and that JSON no longer receives an automatic system instruction.

- [ ] **Step 2: Run tests**

Run: `python3 -m unittest discover -s tests -v`

Expected: PASS.

- [ ] **Step 3: Inspect the local interface**

Run: `python3 main.py`

Repeatable browser check:

1. Keep **Обычный текст**, enter `Три коротких пункта.` in **Инструкция свободного формата**, and wait for the status `сохранено`.
2. Select **JSON** and confirm the instruction field is not visible.
3. Select **Обычный текст** again and confirm the field is visible with `Три коротких пункта.` still present.
4. Change a text field, immediately create/select another session, wait 500 ms, and return to the first session; confirm only the first session contains that change.

Expected: all four observations match; metadata of a sent text response contains the appended instruction, while JSON metadata contains only the base system prompt and `response_format`.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/superpowers/plans/2026-09-05-format-instruction.md
git commit -m "docs: explain format instruction"
```
