# Neutral Model Client Naming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the provider adapter from `ask_deepseek` to `ask_model` so its name describes both DeepSeek and Z.AI calls.

**Architecture:** Keep the adapter as a stateless function in `providers.py`. `Agent`, `AgentRegistry`, `ChatServer`, and the Day 3 experiment already receive a neutrally named callable, so only the provider's exported name and `main.py`'s default import change. No API route, payload, or provider-selection behavior changes.

**Tech Stack:** Python 3, OpenAI-compatible SDK, standard-library `unittest`.

---

### Task 1: Rename the provider adapter and its boot-time import

**Files:**
- Modify: `tests/test_providers.py:23`
- Modify: `providers.py:14`
- Modify: `main.py:1,5`

Run every command in this task from `deepseek-api`.

- [ ] **Step 1: Write the failing test**

  In `tests/test_providers.py`, replace the call to `providers.ask_deepseek(payload)` with `providers.ask_model(payload)`. Keep the existing GLM assertion: Z.AI key, Z.AI URL, and returned content must remain unchanged.

- [ ] **Step 2: Run the focused test and verify it fails**

  Run: `python3 -m unittest tests.test_providers.ProviderTests.test_glm_model_uses_zai_endpoint_and_key -v`

  Expected: failure because `providers.ask_model` does not yet exist.

- [ ] **Step 3: Make the minimal production change**

  Rename `def ask_deepseek(payload, **options):` to `def ask_model(payload, **options):` in `providers.py`. In `main.py`, import `ask_model` and use it as the default value of the existing `create_server(..., ask_model=...)` parameter. Do not rename injected parameters elsewhere: they already describe their role correctly.

- [ ] **Step 4: Run the focused test and verify it passes**

  Run: `python3 -m unittest tests.test_providers.ProviderTests.test_glm_model_uses_zai_endpoint_and_key -v`

  Expected: PASS.

- [ ] **Step 5: Run the complete suite**

  Run: `python3 -m unittest discover -s tests -v`

  Expected: all tests pass.

- [ ] **Step 6: Commit the functional rename**

  ```bash
  git add providers.py main.py tests/test_providers.py
  git commit -m "refactor: use neutral model adapter name"
  ```
