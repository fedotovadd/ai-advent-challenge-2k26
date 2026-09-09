# Agent Deletion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user permanently delete any agent and all of its persisted conversation data from the UI.

**Architecture:** `AgentRegistry` will provide an atomic deletion operation and accept an intentionally empty persisted registry. `ChatRequestHandler` will expose it as a same-origin `DELETE` route. The static client will render a cross button for the selected agent, confirm the destructive action, update selection and clear local data, and reject late responses for deleted agents.

**Tech Stack:** Python 3 standard library (`unittest`, `http.server`, JSON persistence), vanilla HTML/CSS/JavaScript.

---

## File structure

- `agent.py` — transactional registry deletion and empty-state restoration.
- `web.py` — `DELETE /api/agents/{id}` endpoint.
- `static/agent-state.js` — pure browser/client-state helpers, also executable in Node tests.
- `static/index.html` — selected-agent cross button, confirmation, DOM rendering, and HTTP wiring.
- `tests/test_agent.py` — registry and persistence tests.
- `tests/test_web.py` — HTTP route, origin, persistence failure, and UI-source behavior tests.
- `tests/test_client_state.py` — executable JavaScript state-transition tests invoked through Node.

### Task 1: Persisted registry deletion

**Files:**
- Modify: `agent.py: AgentRegistry`
- Test: `tests/test_agent.py: AgentRegistryTests`

- [ ] **Step 1: Write failing registry tests**

Add tests that create a second agent, give it a message/settings/metadata, then call `registry.delete("agent-2")`. Assert the returned snapshot is the deleted agent, `registry.get("agent-2") is None`, the surviving agent remains, and the JSON file contains neither the deleted ID nor its message text. Add a test deleting `agent-1`, restarting from the resulting JSON, and asserting `agents() == []`; a newly created agent must use the previous `nextId`.

- [ ] **Step 2: Run the registry tests to verify failure**

Run: `python -m unittest tests.test_agent.AgentRegistryTests -v`

Expected: FAIL because `AgentRegistry.delete` does not exist and the loader rejects empty `agents`.

- [ ] **Step 3: Write the failing persistence-rollback test**

Patch the persistence write used by deletion to raise `PersistenceError`, call `delete("agent-1")`, and assert the exception is raised while `registry.get("agent-1")` and the prior serialized state remain available.

- [ ] **Step 4: Run the rollback test to verify failure**

Run: `python -m unittest tests.test_agent.AgentRegistryTests -v`

Expected: FAIL because deletion either does not exist or mutates memory before saving.

- [ ] **Step 5: Implement the minimum registry behavior**

Add a `delete(agent_id)` method guarded by the registry lock. It returns `None` for an unknown ID. For a known ID, build the post-deletion state without mutating `self._agents`, write that state via a helper that performs the existing temporary-file-and-replace flow, then replace `self._agents` only after the write succeeds. Preserve `_next_id`. Update state validation so `agents: []` is valid when `nextId` is a positive integer; only compute `max(numbers)` when the list is non-empty.

- [ ] **Step 6: Run the registry tests to verify success**

Run: `python -m unittest tests.test_agent.AgentRegistryTests -v`

Expected: PASS.

- [ ] **Step 7: Commit the registry unit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: delete persisted agents"
```

### Task 2: Deletion HTTP endpoint

**Files:**
- Modify: `web.py: ChatRequestHandler`
- Test: `tests/test_web.py: DeepSeekWebTests`

- [ ] **Step 1: Write failing endpoint tests**

Add tests that create `agent-2`, post a message to it, then send `DELETE /api/agents/agent-2`. Assert `200`, body `{"deletedId": "agent-2"}`, `GET /api/agents` only returns agent-1, and the saved JSON lacks the deleted history. Add tests for `404` with `{"error": "Агент не найден."}`, a patched registry save returning the safe `500`, a foreign `Origin` receiving `403`, and a matching `Origin` succeeding.

Add deterministic lost-race tests for both messages and settings: temporarily wrap `registry.get` to return the target agent, signal an event, and wait; start the HTTP message/settings request in a thread; while it is paused after the lookup, delete the agent; release the request and assert it returns the standard 404 rather than a `200` with `null` or an exception.

- [ ] **Step 2: Run endpoint tests to verify failure**

Run: `python -m unittest tests.test_web.DeepSeekWebTests -v`

Expected: FAIL because `DELETE` is not implemented.

- [ ] **Step 3: Implement the minimum endpoint**

Implement `do_DELETE` with the same-origin check used by `POST` and `PUT`. Match exactly `/api/agents/{agent_id}` (four slash-separated parts), call `registry.delete(agent_id)`, map `None` to the standard 404 JSON error, map `PersistenceError` to `_send_storage_error()`, and on success send `200, {"deletedId": agent_id}`. In `_handle_message` and `_handle_settings`, treat a `None` result from the registry as the same standard 404, covering an agent deleted after the preliminary lookup. All unrelated paths remain 404.

- [ ] **Step 4: Run endpoint tests to verify success**

Run: `python -m unittest tests.test_web.DeepSeekWebTests -v`

Expected: PASS.

- [ ] **Step 5: Commit the endpoint**

```bash
git add web.py tests/test_web.py
git commit -m "feat: expose agent deletion API"
```

### Task 3: Testable client-state helpers

**Files:**
- Create: `static/agent-state.js`
- Create: `tests/test_client_state.py`

- [ ] **Step 1: Write failing executable state tests**

Use Python `subprocess.run` to execute a small Node program that imports `static/agent-state.js`. Test a selected middle item chooses its next neighbor; a selected last item chooses the previous neighbor; deleting the final item gives `activeId: null`; and all per-agent draft, pending, status, metadata, and deferred settings keys are removed. Add a test that the update guard rejects a late response for a deleted ID but accepts a live ID.

- [ ] **Step 2: Run the JavaScript state tests to verify failure**

Run: `python -m unittest tests.test_client_state -v`

Expected: FAIL because `static/agent-state.js` does not exist.

- [ ] **Step 3: Implement pure state helpers**

Create `static/agent-state.js` with no DOM or network dependencies. Export browser and CommonJS-compatible helpers such as `removeAgentState(state, agentId)` and `agentExists(state, agentId)`. `removeAgentState` computes fallback selection from the pre-removal list, returns a new state with the agent and all keyed local data removed, clears global metadata if it belongs to the agent, and removes a scheduled settings save/timer for that agent. Keep it deterministic and side-effect-free so browser code and Node tests share the same logic.

- [ ] **Step 4: Run the JavaScript state tests to verify success**

Run: `python -m unittest tests.test_client_state -v`

Expected: PASS.

- [ ] **Step 5: Commit the client-state unit**

```bash
git add static/agent-state.js tests/test_client_state.py
git commit -m "feat: manage deleted agent client state"
```

### Task 4: Client deletion control and state cleanup

**Files:**
- Modify: `web.py: ChatRequestHandler.do_GET`
- Modify: `static/index.html`
- Test: `tests/test_web.py: DeepSeekWebTests`

- [ ] **Step 1: Write failing UI-source tests**

Add an HTTP test that `GET /static/agent-state.js` returns the new JavaScript file with a JavaScript content type. Extend static-page tests to require that script to load, a cross button for the active sidebar agent (with accessible `aria-label`), a `window.confirm` call whose text includes the selected agent name and a warning that deletion cannot be undone, and a `DELETE` request to `/api/agents/`. Require source evidence that message/settings response handlers use `agentExists` before any state, status, metadata, render, or focus effect.

- [ ] **Step 2: Run UI-source tests to verify failure**

Run: `python -m unittest tests.test_web.DeepSeekWebTests.test_page_contains_chat_controls tests.test_web.DeepSeekWebTests.test_static_agent_state_script_is_served -v`

Expected: FAIL because the page has no delete control or deletion logic.

- [ ] **Step 3: Implement the minimal UI behavior**

Serve `/static/agent-state.js` from `do_GET` with `application/javascript; charset=utf-8`, and load it before the inline client script. Render the selected agent as a compact row inside the sidebar with its existing selection button and a visually obvious `×` button. The cross uses `aria-label="Удалить агента"`; it stops click propagation, asks `window.confirm` using the selected name and an irreversible-deletion warning, flushes settings, sends `DELETE`, and leaves state unchanged on an error. On success, if a scheduled settings timer belongs to that agent, call `clearTimeout(state.settingsTimer)` before clearing it; then replace client state with `removeAgentState` and render.

When `activeId` is null, `renderSettings` clears and disables all settings controls, `renderMetadata` shows placeholders, and the composer remains disabled. After every asynchronous message/settings result, first use `agentExists`: if false, do nothing—do not clear/add pending IDs, add errors, update metadata, render, or focus. Keep the existing message, settings, and creation behavior unchanged for live agents.

- [ ] **Step 4: Run UI-source tests to verify success**

Run: `python -m unittest tests.test_web.DeepSeekWebTests.test_page_contains_chat_controls tests.test_web.DeepSeekWebTests.test_static_agent_state_script_is_served -v`

Expected: PASS.

- [ ] **Step 5: Run all client and HTTP tests**

Run: `python -m unittest tests.test_client_state tests.test_web -v`

Expected: PASS with all client, HTTP, and deletion tests.

- [ ] **Step 6: Commit the client unit**

```bash
git add web.py static/index.html tests/test_web.py
git commit -m "feat: add agent deletion control"
```

### Task 5: Final verification

**Files:**
- Verify: `agent.py`, `web.py`, `static/agent-state.js`, `static/index.html`, `tests/`

- [ ] **Step 1: Run final automated verification**

Run: `python -m unittest discover -s tests -v`

Expected: PASS.

- [ ] **Step 2: Inspect the final diff**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only the intended deletion implementation, tests, and documentation are present. The isolated test-state assertions prove that each confirmed deletion removes that agent's messages, settings, and metadata from `data/agents.json` in a running application.
