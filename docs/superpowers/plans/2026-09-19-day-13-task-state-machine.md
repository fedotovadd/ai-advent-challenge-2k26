# Task State Machine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent, formally validated task state machine with Markdown plans, slash-command control, pause/resume, prompt injection, and automatic execution-stage transitions.

**Architecture:** A new pure `task_state.py` module owns the task schema, command parser, finite-state transitions, prompt block, and model-output protocol. `Agent` keeps `context.taskState`, injects it into normal prompts, strips and applies model markers, while `AgentRegistry` owns plan-file persistence and atomic rollback. The existing HTTP message endpoint dispatches task commands locally alongside memory commands; the current chat UI needs only command-help copy.

**Tech Stack:** Python 3, `unittest`, built-in `http.server`, JSON persistence, Markdown files, vanilla HTML/JavaScript.

---

## File structure

| File | Responsibility |
|---|---|
| `deepseek-api/task_state.py` | Pure task data contracts, command parsing, allowed transitions, model marker parsing, prompt/status rendering. |
| `deepseek-api/tests/test_task_state.py` | Fast unit tests for schema, commands, transitions and marker protocol. |
| `deepseek-api/agent.py` | Store/migrate `taskState`; include it in prompt construction; apply model output and coordinate the plan repository through the registry. |
| `deepseek-api/tests/test_agent.py` | Agent and registry prompt, persistence, pause/resume and restart regression tests. |
| `deepseek-api/web.py` | Recognize local task commands and map errors to safe HTTP responses. |
| `deepseek-api/tests/test_web.py` | End-to-end chat-command and Markdown-plan scenarios. |
| `deepseek-api/static/index.html` | Document the new composer commands without adding a new control surface. |
| `deepseek-api/README.md` | Explain the Day 13 workflow and command contract. |
| `docs/day-13/task-state-machine-results.md` | Record the manual/automated demonstration requested by the assignment. |

### Task 1: Pure task-state contracts and marker protocol

**Files:**
- Create: `deepseek-api/task_state.py`
- Create: `deepseek-api/tests/test_task_state.py`

- [ ] **Step 1: Write failing schema and command tests**

Create `tests/test_task_state.py` with tests for a fresh task and exact command parsing:

```python
from task_state import TaskStateError, default_task_state, parse_task_command

def test_task_command_creates_planning_context():
    command = parse_task_command("/task Лендинг продукта")
    task = default_task_state("task-1", command["title"])

    assert command == {"action": "task", "title": "Лендинг продукта"}
    assert task["stage"] == "PLANNING"
    assert task["paused"] is False
    assert task["current"] == "Собрать требования"
    assert task["expectedAction"] == "Продолжайте отвечать на вопросы агента."

def test_task_parser_rejects_missing_and_extra_arguments():
    with pytest.raises(TaskStateError):
        parse_task_command("/task")
    assert parse_task_command("/execute") == {"action": "execute"}
    with pytest.raises(TaskStateError):
        parse_task_command("/execute сейчас")
```

Use `unittest.TestCase` rather than `pytest`, matching the repository. Cover `/task`, `/execute`, `/planning`, `/pause`, `/resume`, `/status`, unrecognized text (`None`), whitespace normalization and every invalid argument form.

- [ ] **Step 2: Run the contract test to verify RED**

Run: `python3 -m unittest tests.test_task_state -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'task_state'`.

- [ ] **Step 3: Implement the minimal task-state module**

Create constants and pure helpers. Keep all task rules outside `agent.py`:

```python
TASK_STAGES = ("PLANNING", "EXECUTION", "VALIDATION", "DONE")
TASK_PLAN_OPEN = "[[TASK_PLAN]]"
TASK_PLAN_CLOSE = "[[/TASK_PLAN]]"
TASK_STEP_DONE = "[[TASK_STEP_DONE]]"
TASK_DONE = "[[TASK_TRANSITION:DONE]]"

class TaskStateError(ValueError):
    pass

def default_task_state(task_id, title): ...
def valid_task_state(value): ...
def parse_task_command(text): ...
def apply_task_command(task, command, next_task_id): ...
def extract_task_plan(answer, task): ...
def apply_execution_markers(task, answer): ...
def task_prompt_block(task, plan_markdown): ...
def task_status(task): ...
```

`default_task_state` produces exactly the schema in the approved spec. Limit a title to a non-empty 1–300 character string and a plan to 1–24 non-empty numbered steps of at most 500 characters; reject bools where integers are expected. `valid_task_state(None)` is true to support old snapshots; otherwise require exactly the canonical keys and internally consistent `step`, `total`, `plan`, `done`, `current`, `expectedAction`, `paused`, and `planPath` values.

Add `nextTaskId` to the containing context contract (positive integer, default `1`), not to `taskState`. `/task` consumes the current sequence number and returns the incremented value with the new task so a completed `task-1` is never reused for the next task.

`apply_task_command` returns `(new_task, message)` without mutation. It enforces that `/execute` requires a parsed plan, `/planning` only works from execution, pause/resume only work in the stated active states, and paused tasks permit only status/resume. `/task` creates a new task only when no task exists or the old task is `DONE`.

`extract_task_plan` accepts one complete container, returns visible Markdown with the markers removed plus numbered steps, and otherwise returns no plan. A revised plan must retain the existing `done` list as its verbatim ordered prefix and have at least one unfinished step; otherwise raise `TaskStateError` and leave state unchanged. `apply_execution_markers` recognizes `TASK_STEP_DONE` and `TASK_DONE` only when the matching marker is the response’s final non-empty line; a marker elsewhere is ordinary visible text. It advances one step, enters `VALIDATION` after the final step, and only permits `TASK_DONE` from validation. Return a transition message such as `Завершён этап EXECUTION. Начат этап VALIDATION.` when state changes.

- [ ] **Step 4: Run contract tests to verify GREEN**

Run: `python3 -m unittest tests.test_task_state -v`

Expected: PASS. Extend the test file first, then validate:

- every allowed and forbidden finite-state transition;
- pause on planning, execution and validation; resume must preserve stage, step and current;
- exact `[[TASK_PLAN]]` extraction, malformed/multiple containers, invalid numbering, and revision-prefix rejection;
- first and final `[[TASK_STEP_DONE]]`, early/unknown `TASK_DONE`, raw marker removal, and readable status/prompt blocks.
- markers embedded before a later non-empty response line must neither be stripped nor change the state;
- sequential `/task` calls after `DONE` must produce `task-1`, `task-2` and distinct plan paths.

- [ ] **Step 5: Commit the pure state machine**

```bash
git add deepseek-api/task_state.py deepseek-api/tests/test_task_state.py
git commit -m "feat: define task state machine"
```

### Task 2: Agent prompt integration, state migration, and atomic plan persistence

**Files:**
- Modify: `deepseek-api/agent.py:16-75, 118-135, 222-660, 765-965, 1036-1515`
- Modify: `deepseek-api/tests/test_agent.py`

- [ ] **Step 1: Write failing agent and registry tests**

Add focused tests in `tests/test_agent.py` for these externally visible behaviors:

```python
def test_task_state_is_injected_after_profile_and_before_memory():
    calls = []
    instance = Agent("agent-1", "Тест", default_settings(), lambda payload, **_: calls.append(payload) or "Ответ")
    instance.apply_task_command({"action": "task", "title": "Лендинг"})
    instance.respond("Нужен сайт", profile=profile)
    system = calls[-1]["messages"][0]["content"]
    self.assertLess(system.index("[Профиль пользователя]"), system.index("[TASK_STATE]"))
    self.assertLess(system.index("[TASK_STATE]"), system.index(agent.MEMORY_DATA_INSTRUCTION))

def test_registry_restores_paused_execution_without_reexplaining(tmp_path):
    registry = AgentRegistry(model, tmp_path / "agents.json")
    # create task, persist plan, execute it, pause it, restart registry
    restored = AgentRegistry(model, tmp_path / "agents.json").get("agent-1").snapshot()
    self.assertEqual(restored["context"]["taskState"]["stage"], "EXECUTION")
    self.assertTrue(restored["context"]["taskState"]["paused"])
```

Use `TemporaryDirectory` and the registry state path so the plan directory is isolated too. Include an answer containing a plan container, then assert both the visible assistant message and the saved `task-plans/agent-1-task-1.md` text contain Markdown but not raw container delimiters.

- [ ] **Step 2: Run Agent tests to verify RED**

Run: `python3 -m unittest tests.test_agent -v`

Expected: FAIL because `context.taskState`, task-command handling, task prompt injection and plan persistence do not exist.

- [ ] **Step 3: Add state data and plan-store seam to `Agent`**

1. Import the pure task-state helpers. Bump `STATE_VERSION` from 8 to 9 and add `"taskState": None, "nextTaskId": 1` to `default_context()`.
2. Exclude both `taskState` and `nextTaskId` from `_branch_state()` exactly as `memoryLayers` is excluded: task state and its sequence are agent-global, must survive conversation-branch switches, and cannot be restored from stale branch/checkpoint snapshots.
3. Extend `_valid_context()` to validate `context["taskState"]` and positive `nextTaskId`; make migrations backfill both `None` and `1` for v8 snapshots before final validation. Apply that backfill to root contexts and all branch/checkpoint snapshots that carry old metadata, preserving their existing schema. Keep all legacy version migrations intact and change only their final state version to 9 where appropriate.
4. Add `Agent.apply_task_command(command, task_id)` that calls the pure transition helper under the existing lock and returns its local status message. It must never call the model or append chat history.
5. Add `_task_state_messages(system_prompt, context, plan_markdown)` or incorporate `task_prompt_block` into `_memory_layer_messages`, preserving the exact order: base prompt → active user profile → `[TASK_STATE]` → memory data → existing summary/facts/history → current user message. The task block is absent when `taskState` is null.
6. Give `AgentRegistry` a `TaskPlanStore` rooted at `state_path.parent / "task-plans"`, then inject its transaction into `Agent.respond`. `TaskPlanStore.replace(agent_id, task_id, markdown)` must write a temp file and atomically replace `{agent-id}-{task-id}.md`, while recording the old bytes (or that the file was absent). Reads resolve only the stored basename beneath that root, never a user-supplied absolute path.
7. After a successful provider response but before appending the assistant message, extract planning containers or execution markers. Append only visible response text. When a marker changes stages, append the returned Russian transition notice to that same visible assistant message so it is persisted and shown in chat. The existing request payload remains the trace of the sent task block; do not add a mandatory `metadata.taskState` field, avoiding an unnecessary metadata migration. If an invalid model marker/container appears, keep the plain visible answer but do not mutate state or write a plan; do not turn a provider success into an API error.
8. When state is paused, make `Agent.respond` raise `TaskStateError` before payload construction, history mutation or model call. Rebuild `full_history_payload` with the same task system block so context estimates remain comparable.

- [ ] **Step 4: Add registry-level atomicity and persistence**

Extend `AgentRegistry` with `apply_task_command(agent_id, command)`. Follow the existing `apply_memory_command` rollback pattern: snapshot agents before mutation, persist `agents.json`, then restore the snapshot on `PersistenceError`. Create one `TaskPlanStore` transaction per `respond`: for a created/revised plan write the file first, persist state second, and on any failure restore the agent snapshot and call transaction rollback (delete a newly-created plan or atomically restore bytes of a replaced plan). Commit the plan transaction only after `_save()` succeeds. Ensure direct and loaded `Agent` construction receive the same plan-store dependency. Do not alter shared long-term-memory synchronization.

- [ ] **Step 5: Run Agent tests to verify GREEN**

Run: `python3 -m unittest tests.test_task_state tests.test_agent -v`

Expected: PASS. Confirm all current profile, memory, branching and migration tests remain green as well as new tests for:

- a normal planning answer creates/updates a Markdown plan but remains `PLANNING` and expects `/execute`;
- `/execute` and `/planning` preserve `done` and select the next valid plan step;
- `TASK_STEP_DONE` advances first and middle steps, then enters validation only after final completion;
- `TASK_DONE` completes validation and all markers are absent from stored messages;
- automatic stage notices are appended to the visible/persisted assistant message;
- paused state causes zero provider calls, and resumed state sends the same task/step block without repeating task creation;
- v8 root/branch/checkpoint snapshots load with `taskState is None` and `nextTaskId == 1`, malformed v9 task state is rejected safely, and state plus plan survive a registry restart;
- a completed task followed by a new `/task` has a new identifier/file, and injected state-save failure restores both the old snapshot and prior plan file contents.
- switching a context branch after task creation preserves the active task and the monotonic `nextTaskId` rather than restoring a stale branch snapshot.

- [ ] **Step 6: Commit Agent integration**

```bash
git add deepseek-api/agent.py deepseek-api/tests/test_agent.py
git commit -m "feat: persist task state and plans"
```

### Task 3: Local HTTP command dispatch and composer guidance

**Files:**
- Modify: `deepseek-api/web.py:13-22, 286-340`
- Modify: `deepseek-api/tests/test_web.py:1080-1155`
- Modify: `deepseek-api/static/index.html` (existing command-help disclosure)

- [ ] **Step 1: Write failing HTTP/UI regression tests**

Add an end-to-end test that posts local task commands through the existing message route:

```python
status, body, _ = self.json_request(
    "POST", "/api/agents/agent-1/messages", {"text": "/task Статья"},
)
self.assertEqual(status, 200)
self.assertEqual(body["command"]["message"], "Создана задача: Статья.")
self.assertEqual(body["agent"]["messages"], [])
self.assertEqual(self.calls, [])

self.answers = ["[[TASK_PLAN]]\n# План\n1. Исследовать\n[[/TASK_PLAN]]"]
status, body, _ = self.json_request("POST", "/api/agents/agent-1/messages", {"text": "Нужна статья"})
self.assertEqual(status, 200)
self.assertIn("# План", body["agent"]["messages"][-1]["content"])
```

Cover `/execute` without a plan (400), `/planning` outside execution (400), pause on every active stage, blocked normal message while paused (400 and no model call), status while paused (200), resume, unknown agent (404), persistence failure (500), and matching/foreign Origin behavior. Add static assertions that the existing help text contains `/task Название`, `/execute`, `/planning`, `/pause`, `/resume`, and `/status`.

- [ ] **Step 2: Run the web suite to verify RED**

Run: `python3 -m unittest tests.test_web.DeepSeekWebTests -v`

Expected: FAIL because the task command parser is not wired to the endpoint and command help is absent.

- [ ] **Step 3: Dispatch task commands locally in `web.py`**

Import `TaskStateError` and `parse_task_command`. In `_handle_message`, parse memory commands first, then task commands; preserve the current memory-command response contract. For a task command call `registry.apply_task_command` and return the same `200 {agent, command}` shape. Map `TaskStateError` and malformed slash input to `400`, missing agents to `404`, and `PersistenceError` to existing safe `500`. Keep all commands out of provider calls and transcript history.

For ordinary messages, keep the existing `registry.respond` path but catch `TaskStateError` alongside the other client errors. Do not change CORS/origin handling or existing endpoint paths.

- [ ] **Step 4: Update only the composer help text**

In the existing toggleable command-help region of `static/index.html`, add concise Russian documentation:

```text
/task Название — начать опрос и подготовку плана
/execute — подтвердить план и начать выполнение
/planning — вернуться к уточнениям
/pause · /resume · /status — управлять сохранённым состоянием
```

Do not add buttons, dialogs, or a separate state panel; users asked for slash-command control.

- [ ] **Step 5: Run web tests to verify GREEN**

Run: `python3 -m unittest tests.test_web.DeepSeekWebTests -v`

Expected: PASS, including all pre-existing route, metadata, memory-command and same-origin tests.

- [ ] **Step 6: Commit HTTP and help integration**

```bash
git add deepseek-api/web.py deepseek-api/static/index.html deepseek-api/tests/test_web.py
git commit -m "feat: expose task state commands"
```

### Task 4: Documentation and final verification

**Files:**
- Modify: `deepseek-api/README.md`
- Create: `docs/day-13/task-state-machine-results.md`

- [ ] **Step 1: Write the Day 13 usage documentation**

Add a Day 13 section to the README table of contents and body. Explain the exact lifecycle:

1. Send `/task Название` and answer the planning questions.
2. Read the Markdown plan displayed in the chat and saved by the agent.
3. Send `/execute` only to confirm that plan.
4. Use `/planning` if execution reveals a requirement needing clarification.
5. Read automatic transition messages from execution to validation and done.
6. Use `/pause`, restart the server, `/resume`, and `/status` to demonstrate persistence.

List each command’s allowed state and note that commands are local and never sent to the LLM.

- [ ] **Step 2: Record a reproducible result scenario**

Create `docs/day-13/task-state-machine-results.md` with a concise scripted transcript: task creation, planning question, a returned Markdown plan, `/execute`, a completed step, pause/restart/resume, validation, and done. Include the expected state fields at each checkpoint and test commands used; do not include API keys or fabricated live-provider performance data.

- [ ] **Step 3: Run the complete automated suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: exit code 0 with all test modules, including `test_task_state`, passing.

- [ ] **Step 4: Run focused static verification**

Run: `python3 -m py_compile agent.py web.py task_state.py && git diff --check`

Working directory: `deepseek-api` for `py_compile`; repository root for `git diff --check`.

Expected: both commands exit 0 without output.

- [ ] **Step 5: Commit documentation**

```bash
git add deepseek-api/README.md docs/day-13/task-state-machine-results.md
git commit -m "docs: describe task state machine workflow"
```
