# Day 5 Model and Chat Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let each chat select one of three models and retain its own last-request metadata, including response time, token usage, and USD cost.

**Architecture:** Extend the in-memory session schema with a selected model and the last metadata object. The server sends the session model to the provider, normalizes provider usage when available, calculates cost from fixed per-million-token rates, and stores the response metadata in the session. The browser renders the active session’s settings and metadata rather than a global metadata variable.

**Tech Stack:** Python 3 standard-library HTTP server, OpenAI Python client, embedded vanilla JavaScript, unittest.

---

### Task 1: Add per-chat model and metadata server state

**Files:**
- Modify: `deepseek-api/main.py:13-16, 208-267, 600-710`
- Test: `deepseek-api/tests/test_main.py:114-235`

- [ ] **Step 1: Write two focused failing tests**

Add tests that assert: a newly created session includes `settings.model == "deepseek-v4-flash"` and `metadata is None`; and a chat with `glm-4.7-flash` sends that model and returns/stores `responseTimeMs`, usage, and the cost calculated as `(prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000`.

- [ ] **Step 2: Run the two tests to verify RED**

Run: `python3 -m unittest tests.test_main.DeepSeekWebTests.test_initial_session_is_available tests.test_main.DeepSeekWebTests.test_message_uses_session_model_and_stores_usage_metadata`

Expected: failures because neither model nor persisted usage metadata exists.

- [ ] **Step 3: Implement minimal server support**

Add a three-model allowlist and USD-per-million price table. Include `model` in `DEFAULT_SETTINGS`, validate it in `_validate_settings`, and put `metadata: None` in session creation. Use the DeepSeek off-peak cache-miss/output rates (Flash $0.22/$0.66; Pro $0.66/$1.98); mark GLM-4.7-Flash free so it has no numeric price. Change `ask_deepseek()` to return `{"content": response.choices[0].message.content, "usage": response.usage}` after checking content. Update the injected model-call adapter to also accept the current string return or a mapping with `content` and optional `usage`, so existing fake callbacks remain small. Around the provider call, measure elapsed monotonic time, normalize `prompt_tokens`, `completion_tokens`, and `total_tokens`, calculate `costUsd` only when both needed token counts exist, and persist metadata in the matching session on success and provider error.

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run the same unittest command.

Expected: PASS.

### Task 2: Render the active chat’s model and metadata

**Files:**
- Modify: `deepseek-api/main.py:155-205`
- Test: `deepseek-api/tests/test_main.py:92-113`

- [ ] **Step 1: Write one focused failing page test**

Assert the page has `model-input` and the labels for response time, tokens, and cost.

- [ ] **Step 2: Run it to verify RED**

Run: `python3 -m unittest tests.test_main.DeepSeekWebTests.test_page_contains_model_and_usage_metadata`

Expected: FAIL because the controls and labels do not exist.

- [ ] **Step 3: Implement the smallest client change**

Add the model `<select>` with the three requested labels/values. Include it in `elements`, `renderSettings`, and `settingsValues`. Delete the global `state.metadata`; make `renderMetadata` read `activeSession().metadata`; render time in seconds, total tokens, and USD cost with `—` when absent. On new chat and message completion, rely on the returned updated session before `render()`; session switching therefore restores that chat’s own data automatically.

- [ ] **Step 4: Run the focused page test to verify GREEN**

Run the same unittest command.

Expected: PASS.

### Task 3: Verify the prototype

**Files:**
- Modify: `deepseek-api/README.md`
- Test: `deepseek-api/tests/test_main.py`

- [ ] **Step 1: Document the three models, calculation formula, rates, and per-chat persistence**

Add a concise Day 5 section to the README, linking the cited price pages and stating rates are per one million tokens.

- [ ] **Step 2: Run the full existing test suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: PASS.

- [ ] **Step 3: Commit the implementation**

Run: `git add deepseek-api/main.py deepseek-api/tests/test_main.py deepseek-api/README.md && git commit -m "feat: add model choice and per-chat metadata"`
