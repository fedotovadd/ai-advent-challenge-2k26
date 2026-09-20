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
- Modify: `README.md` — link the Day 14 result from the project task list.

### Task 1: Define the invariant contract

**Files:**
- Create: `deepseek-api/tests/test_invariants.py`
- Create: `deepseek-api/invariants.py`

- [ ] **Step 1: Write failing command and schema tests**

```python
class InvariantCommandTests(unittest.TestCase):
    def test_free_form_add_and_local_commands(self):
        self.assertEqual(parse_invariant_command("/invariant Не использовать Python"), {
            "action": "add", "text": "Не использовать Python"
        })
        self.assertEqual(parse_invariant_command("/invariants"), {"action": "list"})
        self.assertEqual(parse_invariant_command("/remove-invariant 2"), {"action": "remove", "number": 2})
        self.assertEqual(parse_invariant_command("/clear-invariants"), {"action": "clear"})

    def test_schema_normalizes_and_rejects_duplicate_or_oversized_values(self):
        self.assertTrue(valid_invariants(["Только Kotlin"]))
        self.assertFalse(valid_invariants(["Kotlin", "kotlin"]))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd deepseek-api && python3 -m unittest tests.test_invariants -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'invariants'`.

- [ ] **Step 3: Implement the smallest standalone contract**

Create constants `MAX_INVARIANTS = 24` and `MAX_INVARIANT_LENGTH = 500`; add `InvariantCommandError`, `normalize_invariant`, `valid_invariants`, `parse_invariant_command`, and `apply_invariant_command`. Keep command output as a plain `{invariants, message}` result and reject invalid syntax before mutation.

- [ ] **Step 4: Run the unit tests**

Run: `cd deepseek-api && python3 -m unittest tests.test_invariants -v`

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
class InvariantConstraintTests(unittest.TestCase):
    def test_stack_rules_reject_python_and_require_kotlin_and_ktor(self):
        rules = ["Использовать Kotlin и Ktor", "Не использовать Python"]
        self.assertIn("Python", violations_for_solution("Реализуйте на Python", rules))
        self.assertIn("Ktor", violations_for_solution("Реализуйте на Kotlin", rules))
        self.assertTrue(conflict_for_addition(["Только Kotlin"], "Использовать Java"))

    def test_prompt_block_marks_rules_as_higher_priority_than_context(self):
        block = invariant_prompt_block(["Только Kotlin"])
        self.assertIn("[ИНВАРИАНТЫ]", block)
        self.assertIn("высшего приоритета", block)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd deepseek-api && python3 -m unittest tests.test_invariants.InvariantConstraintTests -v`

Expected: FAIL because constraint helpers do not exist.

- [ ] **Step 3: Implement transparent constraint helpers**

Parse `использовать`, `только`, `не использовать`, and `без` case-insensitively. Split technology lists only on commas and the standalone word `и`; compare whole normalised terms. Represent positive requirements, one optional allowlist from `только`, and banned terms. Return violations only for an explicit solution recommendation (action phrase plus technology); build the Russian refusal with the original violated rules. Reject contradictory add operations without changing the list.

- [ ] **Step 4: Run the focused tests**

Run: `cd deepseek-api && python3 -m unittest tests.test_invariants -v`

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
def test_invariants_are_agent_global_across_branch_switches(self):
    registry = AgentRegistry(lambda payload, **options: "Ответ", self.state_path)
    registry.apply_invariant_command("agent-1", {"action": "add", "text": "Только Kotlin"})
    registry.update_settings("agent-1", {**default_settings(), "contextStrategy": "branching"})
    checkpoint = registry.context_action("agent-1", "checkpoint", {"name": "до ветки"})
    branch = registry.context_action("agent-1", "branch", {"checkpointId": checkpoint["checkpointId"], "name": "ветка"})
    registry.context_action("agent-1", "switch", {"branchId": branch["branchId"]})
    self.assertEqual(registry.get("agent-1").snapshot()["context"]["invariants"], ["Только Kotlin"])

def test_prompt_places_invariants_after_base_prompt_and_before_profile_and_task(self):
    # Capture payload and assert base prompt < [ИНВАРИАНТЫ] < profile/task text.
    ...

def test_registry_migrates_version_nine_state_without_invariants(self):
    # Write a valid v9 JSON fixture, instantiate AgentRegistry, and assert a v10
    # snapshot with context["invariants"] == [].
    ...

def test_registry_recovers_from_malformed_version_ten_invariants(self):
    # Persist a v10 state with a non-list invariant value and assert safe default recovery.
    ...

def test_invariants_survive_registry_restart_and_rollback_after_save_failure(self):
    # Restart after a successful add; then patch _save_state to raise OSError and
    # assert the attempted command did not change the in-memory array.
    ...
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd deepseek-api && python3 -m unittest tests.test_agent.AgentTests.test_invariants_are_agent_global_across_branch_switches tests.test_agent.AgentTests.test_registry_migrates_version_nine_state_without_invariants tests.test_agent.AgentTests.test_registry_recovers_from_malformed_version_ten_invariants tests.test_agent.AgentTests.test_invariants_survive_registry_restart_and_rollback_after_save_failure -v`

Expected: FAIL because the registry exposes no invariant operation.

- [ ] **Step 3: Implement persistence and injection**

Add `invariants: []` to `default_context`, exclude it from `_save_active_branch` and branch-state restoration, and include `valid_invariants` in `_valid_context`. Raise `STATE_VERSION` to 10 and migrate v9 root contexts by supplying `invariants: []` before validation. Import invariant helpers, add `Agent.apply_invariant_command`, then add an atomic `AgentRegistry.apply_invariant_command` using the existing `_snapshots()` / `_restore(before, shared_long_term)` seam around `_save()`. Append the invariant prompt block immediately after the base system prompt and before `profile_prompt_block` and `task_prompt_block`.

- [ ] **Step 4: Run focused tests**

Run: `cd deepseek-api && python3 -m unittest tests.test_agent.AgentTests -v`

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

def test_positive_and_only_stack_rules_filter_explicit_solutions(self):
    # Verify request and model-response rejection for missing Ktor under
    # «Использовать Kotlin и Ktor» and Java under «Только Kotlin».
    ...

def test_question_about_rule_is_not_an_explicit_solution_request(self):
    # «Почему Python запрещён?» must still reach the provider.
    ...
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd deepseek-api && python3 -m unittest tests.test_agent.AgentTests.test_conflicting_user_request_does_not_call_provider_or_change_history tests.test_agent.AgentTests.test_model_response_with_forbidden_solution_is_replaced_by_refusal tests.test_agent.AgentTests.test_positive_and_only_stack_rules_filter_explicit_solutions tests.test_agent.AgentTests.test_question_about_rule_is_not_an_explicit_solution_request -v`

Expected: FAIL because `respond` still calls the provider and stores the raw answer.

- [ ] **Step 3: Implement the two validation gates**

Add a dedicated `InvariantViolationError` carrying the prebuilt refusal. At the start of `Agent.respond`, after text normalisation but before `_summary_candidate`, `_update_facts`, metrics, payload building, or message mutation, raise it for an explicit conflicting request. After a successful provider response but before task markers and assistant-message persistence, replace a violating answer with the refusal template. Preserve the user message for post-generation filtering; do not run the refusal template through the checker.

- [ ] **Step 4: Run focused tests**

Run: `cd deepseek-api && python3 -m unittest tests.test_agent.AgentTests -v`

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

def test_invariant_command_endpoint_reports_list_mutation_and_errors(self):
    # Exercise /invariants, /remove-invariant 1, /clear-invariants, invalid or
    # out-of-range number, blank /invariant, unknown /remove-invariant syntax,
    # and a contradictory add; assert each error is HTTP 400 {"error": ...}.
    ...

def test_invariant_command_save_failure_rolls_back_and_never_calls_provider(self):
    # Inject persistence failure, assert the original invariants remain, a 500 is
    # returned, and the provider call counter is still zero for every local command.
    ...
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd deepseek-api && python3 -m unittest tests.test_web -v`

Expected: FAIL because the handler does not parse invariant commands or catch invariant violations.

- [ ] **Step 3: Implement endpoint handling and command help**

In `_handle_message`, parse invariant commands before task commands, call the registry method, and return `{agent, command, invariants}`. Catch blank, malformed, out-of-range and contradictory commands as HTTP 400 `{error}`. Preserve the existing storage-error mapping when an atomic local command cannot persist. Catch `InvariantViolationError` around `registry.respond` and return HTTP 400 `{agent: snapshot, error: refusal, accepted: false}`. Add the four invariant commands to the existing “Команды памяти и задачи” panel; no client-state code changes are needed because it already renders non-OK `body.error`.

- [ ] **Step 4: Run HTTP and static tests**

Run: `cd deepseek-api && python3 -m unittest tests.test_web tests.test_memory_ui -v`

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
- Modify: `README.md`

- [ ] **Step 1: Write documentation expectations**

Document the four commands, independent persistence, prompt order, the `Не использовать Python` versus `Реализуй на Python` conflict, and the refusal behaviour. Link Day 14 from the project README.

- [ ] **Step 2: Write documentation**

Add concise Russian usage examples and state the deliberate scope of the deterministic checker: it filters explicit stack recommendations while natural-language business rules are retained as high-priority prompt constraints.

- [ ] **Step 3: Run all automated checks**

Run: `cd deepseek-api && python3 -m unittest discover -s tests -v`

Expected: PASS with no failures.

- [ ] **Step 4: Inspect final diff and working tree**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only the planned Day 14 changes before commit.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md deepseek-api/README.md docs/day-14/invariants-results.md
git commit -m "docs: describe agent invariants"
```
