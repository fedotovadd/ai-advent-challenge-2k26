# Controlled Task Transitions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce an explicit persisted task-state graph so the assistant cannot skip planning or validation, and show a helpful in-chat refusal for every rejected transition.

**Architecture:** Keep `TaskContext` as the persisted source of truth in `context.taskState`. `task_state.py` owns the graph, guards, transition result and recognition/removal of model markers; `Agent.respond()` consumes that result, persists a canonical plan, and appends the user-facing outcome. The web layer remains an HTTP transport around `AgentRegistry`.

**Tech Stack:** Python 3.9+, standard-library `unittest`, local `http.server`, existing task persistence.

---

## File structure

- Modify: `deepseek-api/task_state.py` — declared graph, guarded stage transition API, marker parsing, safe state updates and prompt instructions.
- Modify: `deepseek-api/agent.py` — preserve transition refusals from model output and persist the returned task state/plan.
- Modify: `deepseek-api/tests/test_task_state.py` — unit coverage for every graph edge, rejection, marker suppression and rework reset.
- Modify: `deepseek-api/tests/test_agent.py` — regression test for an in-chat refusal and saved state across restart.
- Modify: `deepseek-api/tests/test_web.py` — end-to-end task response and continuation coverage through HTTP.
- Modify: `deepseek-api/README.md` — document the Day 15 lifecycle, validation pass/fail markers and pause/resume semantics.
- Modify: `README.md` — add the Day 15 assignment link.
- Create: `docs/day-15/controlled-transitions-results.md` — concise deliverable and manual verification scenarios.

### Task 1: Make `task_state.py` the single owner of the lifecycle graph

**Files:**
- Modify: `deepseek-api/task_state.py:6-230`
- Test: `deepseek-api/tests/test_task_state.py:1-105`

- [ ] **Step 1: Write failing unit tests for guarded transitions.**

Add tests that import the graph/transition API and assert that it permits only `PLANNING → EXECUTION`, `EXECUTION → PLANNING|VALIDATION`, and `VALIDATION → EXECUTION|DONE`; assert that `/execute` without a saved plan has the exact explanatory refusal and that invalid requests leave a deep-equal state.

- [ ] **Step 2: Run the new tests to verify they fail.**

Run: `../.venv/bin/python -m unittest tests.test_task_state.TaskStateTests.test_guarded_transition_contract -v`

Expected: FAIL because the declared graph and transition API do not yet exist.

- [ ] **Step 3: Implement the declared graph and guard API.**

Add a `TASK_TRANSITIONS` mapping and one internal/public transition function that deep-copies the task, checks the mapping before assigning `stage`, and applies per-edge guards. Route `/execute`, `/planning`, final-step completion and validation completion through it. Keep pause/resume as non-stage operations and preserve `stage`, `step`, `current` and `expectedAction` exactly on `/resume`.

- [ ] **Step 4: Run the focused transition test.**

Run: `../.venv/bin/python -m unittest tests.test_task_state.TaskStateTests.test_guarded_transition_contract -v`

Expected: PASS.

- [ ] **Step 5: Write failing tests for model-marker controls.**

Add tests for `[[TASK_TRANSITION:DONE]]` from `EXECUTION`, `[[TASK_STEP_DONE]]` from `PLANNING`, `[[TASK_VALIDATION_PASSED]]` from `EXECUTION`, and a marker not in the final nonempty line. Each must remove the technical marker, retain the original state, and return an explanatory `notice`. Add a success path for `[[TASK_VALIDATION_PASSED]]` only in `VALIDATION`.

- [ ] **Step 6: Run the new marker tests to verify they fail.**

Run: `../.venv/bin/python -m unittest tests.test_task_state.TaskStateTests.test_rejected_markers_are_hidden_and_explained -v`

Expected: FAIL because current code exposes rejected/non-final markers and accepts the old final marker.

- [ ] **Step 7: Implement safe marker extraction and validation outcomes.**

Declare `TASK_VALIDATION_PASSED` and a bounded parser for `[[TASK_VALIDATION_FAILED:<number>]]`. Strip every recognized marker from the visible text. Only one marker on the final nonempty line may request an action; otherwise return a no-change refusal. Make `TASK_VALIDATION_PASSED` the sole `VALIDATION → DONE` signal. For a valid failure marker, transition to `EXECUTION`, truncate `done` before its 1-based step, set `step`, `current` and `expectedAction`, and retain a valid completed prefix. Return a uniform result containing the copied task, visible text, `notice`, `continueTask`, acceptance state and next-action guidance.

- [ ] **Step 8: Run all task-state tests.**

Run: `../.venv/bin/python -m unittest tests.test_task_state -v`

Expected: PASS, including existing planning, pause, checklist and state-schema cases.

- [ ] **Step 9: Commit the state-machine contract.**

```bash
git add deepseek-api/task_state.py deepseek-api/tests/test_task_state.py
git commit -m "feat: enforce task state transitions"
```

### Task 2: Surface controlled refusals through the agent and persist rework

**Files:**
- Modify: `deepseek-api/agent.py:580-750`
- Test: `deepseek-api/tests/test_agent.py:90-115, 600-635`
- Test: `deepseek-api/tests/test_web.py:1155-1220`

- [ ] **Step 1: Write a failing agent regression test.**

Create a planned/executing task with a fake provider returning `[[TASK_TRANSITION:DONE]]`. Assert the saved assistant message omits the marker, includes the lifecycle explanation, leaves stage/step/done unchanged, and remains unchanged after constructing a new `AgentRegistry` from the saved state.

- [ ] **Step 2: Run the agent regression test to verify it fails.**

Run: `../.venv/bin/python -m unittest tests.test_agent.AgentRegistryTests.test_invalid_task_transition_is_explained_and_persisted -v`

Expected: FAIL because `Agent.respond()` currently swallows `TaskStateError` and the state layer returns the raw marker.

- [ ] **Step 3: Integrate the transition result in `Agent.respond()`.**

Separate plan-extraction errors from marker handling so only malformed optional plans are ignored. Always consume the state-machine marker result, write the canonical plan after accepted plan/step/rework changes, append `notice` to the visible assistant response, and derive `_last_task_continue` solely from the returned contract. Do not alter history or task state when the result is rejected.

- [ ] **Step 4: Run the agent regression test.**

Run: `../.venv/bin/python -m unittest tests.test_agent.AgentRegistryTests.test_invalid_task_transition_is_explained_and_persisted -v`

Expected: PASS.

- [ ] **Step 5: Write failing HTTP-flow tests.**

Extend the existing task tests with a full flow: validate that `DONE` during execution is visible as a refusal, complete the final execution step, return `[[TASK_VALIDATION_FAILED:2]]`, confirm the task resumes at step 2 after `/pause` and `/resume`, then complete the remaining step and return `[[TASK_VALIDATION_PASSED]]` to reach `DONE`.

- [ ] **Step 6: Run the HTTP-flow test to verify it fails.**

Run: `../.venv/bin/python -m unittest tests.test_web.DeepSeekWebTests.test_task_rejects_skips_and_resumes_rework -v`

Expected: FAIL because validation-rework and validation-pass markers are not implemented.

- [ ] **Step 7: Make the smallest integration changes needed for the HTTP flow.**

Keep routes and response schemas unchanged. Ensure `/pause` blocks automatic continuation, `/resume` restores the state-machine-selected step, and `/api/agents/<id>/task/advance` continues only when the transition result sets `continueTask`.

- [ ] **Step 8: Run focused agent and web suites.**

Run: `../.venv/bin/python -m unittest tests.test_agent tests.test_web -v`

Expected: PASS.

- [ ] **Step 9: Commit the agent integration.**

```bash
git add deepseek-api/agent.py deepseek-api/tests/test_agent.py deepseek-api/tests/test_web.py
git commit -m "feat: explain rejected task transitions"
```

### Task 3: Document the Day 15 lifecycle and verify the release

**Files:**
- Modify: `deepseek-api/README.md:16-46, 420-478`
- Modify: `README.md:5-8`
- Create: `docs/day-15/controlled-transitions-results.md`

- [ ] **Step 1: Update user-facing documentation.**

Add Day 15 to both tables of contents. Document the allowed graph, the requirement for a saved plan before `/execute`, the distinct validation-pass and validation-failure markers, the no-change refusal behavior, and how pause/resume preserves the current step. Create the Day 15 result note with concrete manual checks for skipping execution, skipping validation, rework, and restart/resume.

- [ ] **Step 2: Run documentation and source consistency checks.**

Run: `git diff --check && rg -n "TASK_VALIDATION_PASSED|TASK_VALIDATION_FAILED|День 15" README.md deepseek-api/README.md docs/day-15 deepseek-api/task_state.py`

Expected: no whitespace errors and every documented marker present in the implementation.

- [ ] **Step 3: Run the complete test suite.**

Run: `../.venv/bin/python -m unittest discover -s tests -v`

Expected: PASS. This command needs the sandbox permission that allows test-local sockets on `127.0.0.1`.

- [ ] **Step 4: Inspect the final change set.**

Run: `git status --short && git diff --check && git log --oneline day-14..HEAD`

Expected: only the Day 15 implementation, tests and documentation; no whitespace errors.

- [ ] **Step 5: Commit the documentation.**

```bash
git add README.md deepseek-api/README.md docs/day-15/controlled-transitions-results.md
git commit -m "docs: document controlled task lifecycle"
```
