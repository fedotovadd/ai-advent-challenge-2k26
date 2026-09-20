# Agent Invariants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every agent a persistent invariant layer, enforce explicit stack restrictions before and after LLM calls, and explain conflicts safely.

**Architecture:** `invariants.py` owns the bounded data contract, command grammar, constraint parsing, prompt block, and refusal decisions. `Agent` persists agent-global rules and uses the module before facts/summary/model calls and after a model response. `ChatRequestHandler` treats invariant commands like existing local memory commands and returns the defined refusal shape.

**Tech Stack:** Python 3 standard library, `unittest`, existing `ThreadingHTTPServer` UI.

---

## File structure

- Create: `deepseek-api/invariants.py` — invariant schema, commands, normalisation, parsed stack constraints, prompt/refusal helpers.
- Modify: `deepseek-api/agent.py` — context persistence/migration, prompt injection, request/response enforcement, registry mutation and rollback.
- Modify: `deepseek-api/web.py` — local command dispatch plus HTTP 400 early-refusal response.
- Modify: `deepseek-api/static/index.html` — document chat commands in the existing command panel.
- Create: `deepseek-api/tests/test_invariants.py` — unit contracts for invariant data and parsing.
- Modify: `deepseek-api/tests/test_agent.py` — persistence, prompt order, provider-free conflicts, output filtering and rollback.
- Modify: `deepseek-api/tests/test_web.py` — HTTP command and refusal contracts.
- Create: `docs/day-14/invariants-results.md` — usage examples and conflict-test results.

### Task 1: Define the invariant contract

**Files:**
- Create: `deepseek-api/tests/test_invariants.py`
- Create: `deepseek-api/invariants.py`

- [ ] **Step 1: Write failing command and schema tests**

```python
from invariants import InvariantCommandError, parse_invariant_command, valid_invariants

def test_free_form_add_and_local_commands():
    assert parse_invariant_command("/invariant Не использовать Python") == {
        "action": "add", "text": "Не использовать Python"
    }
    assert parse_invariant_command("/invariants") == {"action": "list"}
    assert parse_invariant_command("/remove-invariant 2") == {"action": "remove", "number": 2}
    assert parse_invariant_command("/clear-invariants") == {"action": "clear"}

def test_schema_normalizes_and_rejects_duplicate_or_oversized_values():
    assert valid_invariants(["Только Kotlin"])
    assert not valid_invariants(["Kotlin", "kotlin"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest tests.test_invariants -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'invariants'`.

- [ ] **Step 3: Implement the smallest standalone contract**

Create constants `MAX_INVARIANTS = 24` and `MAX_INVARIANT_LENGTH = 500`; add `InvariantCommandError`, `normalize_invariant`, `valid_invariants`, `parse_invariant_command`, and `apply_invariant_command`. Keep command output as a plain `{invariants, message}` result and reject invalid syntax before mutation.

- [ ] **Step 4: Run the unit tests**

Run: `python3 -m unittest tests.test_invariants -v`

Expected: PASS.

- [ ] **Step 5: Commit the contract**

```bash
git add deepseek-api/invariants.py deepseek-api/tests/test_invariants.py
git commit -m "feat: add invariant command contract"
```

### Task 2: Parse constraints and generate safe invariant text

**Files:**
- Modify: `deepseek-api/tests/test_invariants.py`
- Modify: `deepseek-api/invariants.py`

- [ ] **Step 1: Write failing constraint tests**

```python
from invariants import conflict_for_addition, invariant_prompt_block, violations_for_solution

def test_stack_rules_reject_python_and_require_kotlin_and_ktor():
    rules = ["Использовать Kotlin и Ktor", "Не использовать Python"]
    assert "Python" in violations_for_solution("Реализуйте на Python", rules)
    assert "Ktor" in violations_for_solution("Реализуйте на Kotlin", rules)
    assert conflict_for_addition(["Только Kotlin"], "Использовать Java")

def test_prompt_block_marks_rules_as_higher_priority_than_context():
    block = invariant_prompt_block(["Только Kotlin"])
    assert "[ИНВАРИАНТЫ]" in block
    assert "высшего приоритета" in block
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest tests.test_invariants.InvariantConstraintTests -v`

Expected: FAIL because constraint helpers do not exist.

- [ ] **Step 3: Implement transparent constraint helpers**

Parse `использовать`, `только`, `не использовать`, and `без` case-insensitively. Split technology lists only on commas and the standalone word `и`; compare whole normalised terms. Represent positive requirements, one optional allowlist from `только`, and banned terms. Return violations only for an explicit solution recommendation (action phrase plus technology); build the Russian refusal with the original violated rules. Reject contradictory add operations without changing the list.

- [ ] **Step 4: Run the focused tests**

Run: `python3 -m unittest tests.test_invariants -v`

Expected: PASS.

- [ ] **Step 5: Commit constraint support**

```bash
git add deepseek-api/invariants.py deepseek-api/tests/test_invariants.py
git commit -m "feat: validate stack invariants"
```

### Task 3: Persist agent-global invariants and inject them into prompts

**Files:**
- Modify: `deepseek-api/tests/test_agent.py`
- Modify: `deepseek-api/agent.py`

- [ ] **Step 1: Write failing agent tests**

```python
def test_invariants_are_agent_global_and_restore_from_version_nine_state(self):
    registry = AgentRegistry(lambda payload, **options: "Ответ", self.state_path)
    registry.apply_invariant_command("agent-1", {"action": "add", "text": "Только Kotlin"})
    registry.context_action("agent-1", "branch", {"checkpointId": None, "name": "ветка"})
    self.assertEqual(registry.get("agent-1").snapshot()["context"]["invariants"], ["Только Kotlin"])

def test_prompt_places_invariants_after_base_prompt_and_before_profile_and_task(self):
    # Capture payload and assert base prompt < [ИНВАРИАНТЫ] < profile/task text.
    ...
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest tests.test_agent.AgentTests.test_invariants_are_agent_global_and_restore_from_version_nine_state -v`

Expected: FAIL because the registry exposes no invariant operation.

- [ ] **Step 3: Implement persistence and injection**

Add `invariants: []` to `default_context`, exclude it from `_save_active_branch` and branch-state restoration, and include `valid_invariants` in `_valid_context`. Raise `STATE_VERSION` to 10 and migrate v9 root contexts by supplying `invariants: []` before validation. Import invariant helpers, add `Agent.apply_invariant_command`, then add an atomic `AgentRegistry.apply_invariant_command` that restores the prior list if `_persist()` fails. Append the invariant prompt block immediately after the base system prompt and before `profile_prompt_block` and `task_prompt_block`.

- [ ] **Step 4: Run focused tests**

Run: `python3 -m unittest tests.test_agent.AgentTests -v`

Expected: PASS.

- [ ] **Step 5: Commit persistence**

```bash
git add deepseek-api/agent.py deepseek-api/tests/test_agent.py
git commit -m "feat: persist agent invariants"
```

### Task 4: Enforce invariants before and after LLM generation

**Files:**
- Modify: `deepseek-api/tests/test_agent.py`
- Modify: `deepseek-api/agent.py`

- [ ] **Step 1: Write failing enforcement tests**

```python
def test_conflicting_user_request_does_not_call_provider_or_change_history(self):
    calls = []
    instance = Agent("agent-1", "Тест", default_settings(), lambda *args, **kwargs: calls.append(1))
    instance.apply_invariant_command({"action": "add", "text": "Не использовать Python"})
    with self.assertRaisesRegex(InvariantViolationError, "Python"):
        instance.respond("Реализуй на Python")
    self.assertEqual(calls, [])
    self.assertEqual(instance.snapshot()["messages"], [])

def test_model_response_with_forbidden_solution_is_replaced_by_refusal(self):
    instance = Agent("agent-1", "Тест", default_settings(), lambda *_args, **_kwargs: "Используйте Python")
    instance.apply_invariant_command({"action": "add", "text": "Не использовать Python"})
    result = instance.respond("Предложи решение")
    self.assertIn("не может предложить", result["messages"][-1]["content"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_agent.AgentTests.test_conflicting_user_request_does_not_call_provider_or_change_history tests.test_agent.AgentTests.test_model_response_with_forbidden_solution_is_replaced_by_refusal -v`

Expected: FAIL because `respond` still calls the provider and stores the raw answer.

- [ ] **Step 3: Implement the two validation gates**

Add a dedicated `InvariantViolationError` carrying the prebuilt refusal. At the start of `Agent.respond`, after text normalisation but before `_summary_candidate`, `_update_facts`, metrics, payload building, or message mutation, raise it for an explicit conflicting request. After a successful provider response but before task markers and assistant-message persistence, replace a violating answer with the refusal template. Preserve the user message for post-generation filtering; do not run the refusal template through the checker.

- [ ] **Step 4: Run focused tests**

Run: `python3 -m unittest tests.test_agent.AgentTests -v`

Expected: PASS.

- [ ] **Step 5: Commit enforcement**

```bash
git add deepseek-api/agent.py deepseek-api/tests/test_agent.py
git commit -m "feat: enforce agent invariants"
```

### Task 5: Expose commands and refusal over HTTP

**Files:**
- Modify: `deepseek-api/tests/test_web.py`
- Modify: `deepseek-api/web.py`
- Modify: `deepseek-api/static/index.html`

- [ ] **Step 1: Write failing HTTP and UI tests**

```python
def test_invariant_commands_are_local_and_return_current_rules(self):
    status, body = self.request("POST", "/api/agents/agent-1/messages", {"text": "/invariant Не использовать Python"})
    self.assertEqual(status, 200)
    self.assertEqual(body["invariants"], ["Не использовать Python"])

def test_conflicting_message_returns_400_without_history_mutation(self):
    # Add rule, then POST «Реализуй на Python».
    self.assertEqual(status, 400)
    self.assertFalse(body["accepted"])
    self.assertEqual(body["agent"]["messages"], [])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest tests.test_web -v`

Expected: FAIL because the handler does not parse invariant commands or catch invariant violations.

- [ ] **Step 3: Implement endpoint handling and command help**

In `_handle_message`, parse invariant commands before task commands, call the registry method, and return `{agent, command, invariants}`. Catch command errors as HTTP 400. Catch `InvariantViolationError` around `registry.respond` and return HTTP 400 `{agent: snapshot, error: refusal, accepted: false}`. Add the four invariant commands to the existing “Команды памяти и задачи” panel; no client-state code changes are needed because it already renders non-OK `body.error`.

- [ ] **Step 4: Run HTTP and static tests**

Run: `python3 -m unittest tests.test_web tests.test_memory_ui -v`

Expected: PASS.

- [ ] **Step 5: Commit endpoint support**

```bash
git add deepseek-api/web.py deepseek-api/static/index.html deepseek-api/tests/test_web.py
git commit -m "feat: expose invariant commands"
```

### Task 6: Document and verify the completed feature

**Files:**
- Create: `docs/day-14/invariants-results.md`
- Modify: `deepseek-api/README.md`

- [ ] **Step 1: Write documentation expectations**

Document the four commands, independent persistence, prompt order, the `Не использовать Python` versus `Реализуй на Python` conflict, and the refusal behaviour. Link Day 14 from the project README.

- [ ] **Step 2: Write documentation**

Add concise Russian usage examples and state the deliberate scope of the deterministic checker: it filters explicit stack recommendations while natural-language business rules are retained as high-priority prompt constraints.

- [ ] **Step 3: Run all automated checks**

Run: `python3 -m unittest discover -s deepseek-api/tests -v`

Expected: PASS with no failures.

- [ ] **Step 4: Inspect final diff and working tree**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only the planned Day 14 changes before commit.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md deepseek-api/README.md docs/day-14/invariants-results.md
git commit -m "docs: describe agent invariants"
```
