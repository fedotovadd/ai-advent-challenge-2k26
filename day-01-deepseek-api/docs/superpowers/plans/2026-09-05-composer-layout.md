# Закреплённое поле сообщения и упрощение настроек Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Закрепить форму отправки сообщения в нижней части чата и удалить настройку свободного формата из приложения.

**Architecture:** Единственная строка HTML в `main.py` содержит стили, разметку и клиентскую логику, поэтому изменения остаются локальными этому файлу. Настройки сессии и их валидация меняются согласованно, чтобы клиент больше не отправлял удалённый ключ. Тесты HTTP остаются контрактом для страницы, API настроек и payload модели.

**Tech Stack:** Python 3, `http.server`, встроенный HTML/CSS/JavaScript, `unittest`.

---

## File structure

- Modify: `day-01-deepseek-api/main.py` — разметка, CSS, клиентский объект настроек, серверная валидация и построение payload.
- Modify: `day-01-deepseek-api/tests/test_main.py` — контракты страницы, настроек и payload без `formatInstruction`.
- Modify: `day-01-deepseek-api/README.md` — описание оставшихся настроек и закреплённой формы.

### Task 1: Зафиксировать контракты удалённой настройки и компоновки чата

**Files:**
- Modify: `day-01-deepseek-api/tests/test_main.py:59-78, 96-171, 213-253`

- [ ] **Step 1: Write failing tests for the new page and settings contract**

  Change page assertions to require the desktop layout markers `height:100vh`, `overflow:hidden`, and `flex:0 0 auto` for both the chat header and composer, plus the narrow-screen override that restores page scrolling between stacked panels. Require `format-instruction-input` and `Инструкция свободного формата` to be absent. Replace the initial and created-session assertions that reference `main.DEFAULT_SETTINGS` with explicit expected settings containing only `systemPrompt`, `format`, `maxTokens`, and `stop`. Update settings request fixtures to contain only those fields; delete tests dedicated to `formatInstruction`. Add a settings request that still contains `formatInstruction` and assert it returns 400. In the text-format model test, assert that the base system prompt is forwarded unchanged.

- [ ] **Step 2: Run the focused test suite to verify it fails**

  Run: `python3 -m unittest discover -s tests -v`

  Expected: failures because page markup and settings validation still include `formatInstruction`, and the CSS lacks the desktop and narrow-screen layout contracts.

- [ ] **Step 3: Commit the red test change**

  ```bash
  git add day-01-deepseek-api/tests/test_main.py
  git commit -m "test: define simplified response controls"
  ```

### Task 2: Удалить настройку и закрепить форму отправки

**Files:**
- Modify: `day-01-deepseek-api/main.py:13-19, 29-43, 49-75, 212-287`
- Test: `day-01-deepseek-api/tests/test_main.py`

- [ ] **Step 1: Implement the smallest change that satisfies the settings contract**

  Remove `formatInstruction` from `DEFAULT_SETTINGS`, the right-panel label and textarea, the JavaScript element map, `renderFormatInstruction`, `renderSettings`, `settingsValues`, and input listener list. Remove the `elif` branch that appends it to the system prompt. Make `_validate_settings` expect and return exactly `systemPrompt`, `format`, `maxTokens`, and `stop`.

- [ ] **Step 2: Implement the fixed composer layout**

  On desktop, make the app and central chat pane use the viewport height. Keep the chat as a vertical flex container with hidden outer overflow; make `header` and `.composer-area` nonshrinking and `.thread` the flex-growing scroll container. At the existing 860px breakpoint, restore normal page scrolling for the stacked sidebar, chat and metadata panels while giving the chat itself the viewport height, so its composer remains visible once the user reaches that panel. Preserve the same 500 ms autosave for non-format settings.

- [ ] **Step 3: Run the focused suite to verify it passes**

  Run: `python3 -m unittest discover -s tests -v`

  Expected: all tests pass, including the revised page, settings and payload assertions.

- [ ] **Step 4: Commit the implementation**

  ```bash
  git add day-01-deepseek-api/main.py day-01-deepseek-api/tests/test_main.py
  git commit -m "feat: simplify response controls"
  ```

### Task 3: Обновить пользовательскую документацию

**Files:**
- Modify: `day-01-deepseek-api/README.md:31-39`

- [ ] **Step 1: Update the settings description**

  Remove the bullet about «Инструкция свободного формата». State that the editable System prompt is sent unchanged for regular text and that JSON still uses only `response_format={"type":"json_object"}`. Mention that the message field remains visible while the conversation scrolls.

- [ ] **Step 2: Run the complete suite again**

  Run: `python3 -m unittest discover -s tests -v`

  Expected: all tests pass.

- [ ] **Step 3: Commit the documentation**

  ```bash
  git add day-01-deepseek-api/README.md
  git commit -m "docs: describe simplified response controls"
  ```
