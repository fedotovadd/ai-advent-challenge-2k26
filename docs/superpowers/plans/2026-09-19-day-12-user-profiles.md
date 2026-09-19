# Day 12 User Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent, application-wide user profiles that automatically personalize each agent reply and can be selected, created, and edited from the chat UI.

**Architecture:** Keep the collection and active-profile pointer in `AgentRegistry`, isolated from per-agent settings and memory. A focused `user_profiles.py` module owns normalization, validation, prompt rendering, and state invariants. The registry passes the active profile only to user-facing `Agent.respond` calls; the server exposes profile routes; a focused browser module manages the picker and its dialogs.

**Tech Stack:** Python 3.14 standard library (`unittest`, `http.server`, JSON persistence), existing vanilla JavaScript UI, OpenAI-compatible provider options.

---

## File structure

- Create: `deepseek-api/user_profiles.py` — profile defaults, validation, ids, prompt block renderer, collection invariants.
- Modify: `deepseek-api/agent.py` — pass a profile into user-facing prompt composition, trace its immutable snapshot, remove the textual JSON prompt instruction, persist and migrate profile collection state.
- Modify: `deepseek-api/web.py` — profile REST routes and strict request parsing/status responses.
- Create: `deepseek-api/static/profile-state.js` — pure browser state transitions for activation/save outcomes and send eligibility.
- Create: `deepseek-api/static/profile-controls.js` — global profile picker, active selection, create and edit dialogs.
- Modify: `deepseek-api/static/index.html` — host the profile section/dialogs, load/mount focused state/control modules, coordinate pending profile saves before chat sends.
- Modify: `deepseek-api/tests/test_agent.py` — unit tests for profile composition, registry CRUD, migration and persistence.
- Modify: `deepseek-api/tests/test_web.py` — HTTP and rendered-page contract tests.
- Modify: `deepseek-api/README.md` — Day 12 usage, data flow, and manual verification example.

### Task 1: Profile domain module

**Files:**
- Create: `deepseek-api/user_profiles.py`
- Modify: `deepseek-api/tests/test_agent.py`

- [ ] **Step 1: Write failing domain tests**

Add focused tests importing `user_profiles` that require a normalized profile, unique names after `casefold()`, a 20-profile limit, and a stable preference block:

```python
def test_profile_normalization_and_preference_block():
    profile = user_profiles.normalize_profile({
        "name": "  Анна  ", "style": " ясно ",
        "format": " абзацы ", "constraints": " без Markdown ",
    })
    self.assertEqual(profile["name"], "Анна")
    self.assertIn("[Профиль пользователя]", user_profiles.profile_prompt_block(profile))
```

- [ ] **Step 2: Run the domain tests and observe RED**

Run: `python3 -m unittest tests.test_agent.AgentRegistryTests.test_profile_normalization_and_preference_block -v`

Expected: FAIL because `user_profiles` does not exist.

- [ ] **Step 3: Implement the smallest standalone profile API**

Create `user_profiles.py` with constants and pure helpers:

```python
MAX_USER_PROFILES = 20
DEFAULT_PROFILE_FIELDS = {"name": "Пользователь", "style": "ясный и дружелюбный", "format": "краткие абзацы", "constraints": "без Markdown"}

def normalize_profile(data): ...
def profile_with_id(profile_id, fields): ...
def valid_profile_collection(profiles, active_profile_id, next_profile_id): ...
def profile_prompt_block(profile): ...
```

Require exactly the editable keys at the HTTP boundary, trim all field values, enforce 1–80/1–500 limits, use `profile-N` ids, and render the labeled block with a non-conflict statement. Never render `id` into the model prompt.

- [ ] **Step 4: Run the focused domain tests and observe GREEN**

Run: `python3 -m unittest tests.test_agent -v`

Expected: all existing agent tests plus new domain tests pass.

- [ ] **Step 5: Commit the domain unit**

```bash
git add deepseek-api/user_profiles.py deepseek-api/tests/test_agent.py
git commit -m "feat: add user profile domain model"
```

### Task 2: Prompt composition and JSON-mode behavior

**Files:**
- Modify: `deepseek-api/agent.py`
- Modify: `deepseek-api/tests/test_agent.py`
- Modify: `deepseek-api/tests/test_web.py`

- [ ] **Step 1: Write failing agent tests**

Add tests that create an agent with non-empty working memory and JSON mode, then call `respond` with a profile. Assert that the system message contains blocks in this exact order: agent system prompt, profile block, memory data. Assert `response_format={"type": "json_object"}` is still sent, while the exact old text `Верни только валидный JSON без Markdown-разметки.` is absent. Assert `metadata["userProfile"]` is a deep copy.

- [ ] **Step 2: Run the focused tests and observe RED**

Run: `python3 -m unittest tests.test_agent.AgentTests.test_respond_applies_profile_before_memory_without_json_prompt_text -v`

Expected: FAIL because `Agent.respond` has no profile argument and metadata lacks `userProfile`.

- [ ] **Step 3: Implement minimal composition changes**

Change `Agent.respond` to accept an optional validated profile supplied by the registry; an isolated direct call obtains `default_profile()` so all existing direct agent/strategy tests remain compatible. Build `system_prompt` from the per-agent value plus `profile_prompt_block(profile)` before `_memory_layer_messages`. Delete `JSON_OUTPUT_INSTRUCTION` and do not append text when JSON mode is enabled; retain only the existing provider option. Add `userProfile` to newly produced metadata, but keep the current persistence validator temporarily compatible with both legacy metadata (no key) and current metadata (profile object); Task 3 will backfill legacy data and make the v8 schema strict. In `tests/test_web.py`, update the existing JSON endpoint expectation to require the provider option and absence of the removed prompt text.

- [ ] **Step 4: Run the agent suite and observe GREEN**

Run: `python3 -m unittest tests.test_agent tests.test_context_strategies tests.test_web.DeepSeekWebTests.test_json_settings_are_sent_to_the_model -v`

Expected: PASS, including old JSON option behavior and new prompt/snapshot assertions.

- [ ] **Step 5: Commit prompt behavior**

```bash
git add deepseek-api/agent.py deepseek-api/tests/test_agent.py deepseek-api/tests/test_web.py
git commit -m "feat: apply profiles to agent replies"
```

### Task 3: Registry profile lifecycle and v7-to-v8 migration

**Files:**
- Modify: `deepseek-api/agent.py`
- Modify: `deepseek-api/tests/test_agent.py`

- [ ] **Step 1: Write failing registry tests**

Cover `profiles()`, `create_profile(fields)`, `update_profile(id, fields)`, and `activate_profile(id)`. Verify a created profile becomes active, updates preserve its id, selection affects replies from two distinct agents, a storage error restores profiles/active id/next id, and a restart preserves all three values.

Build a representative v7 JSON state with branch and checkpoint metadata, load it, and assert migration produces one `profile-1`, sets legacy metadata `userProfile` to `None`, and retains agent data. Add corrupt v8 states with duplicate ids/names, missing active id, or a low `nextProfileId`; assert the registry safely starts from defaults.

- [ ] **Step 2: Run the focused registry tests and observe RED**

Run: `python3 -m unittest tests.test_agent.AgentRegistryTests.test_registry_creates_activates_and_persists_profiles -v`

Expected: FAIL because the registry has no profile lifecycle.

- [ ] **Step 3: Implement registry state and migration**

Bump `STATE_VERSION` to 8. Store `profiles`, `activeProfileId`, and `nextProfileId` in `_state()` and deletion state. On mutations, snapshot all profile state before saving and restore it after `PersistenceError`. Pass the resolved active profile into `Agent.respond` only from `AgentRegistry.respond`.

In `_load()`, migrate v7 before strict v8 validation. Walk current, branch, and checkpoint metadata to insert `userProfile: None` only for historical metadata; then require the `userProfile` field in all v8 metadata while retaining `None` for the historical snapshots. Reject malformed v8 collections through `valid_profile_collection` and fall back to the safe default registry state.

- [ ] **Step 4: Run all backend tests and observe GREEN**

Run: `python3 -m unittest tests.test_agent tests.test_context_strategies -v`

Expected: PASS; unrelated memory and branching behavior remains unchanged.

- [ ] **Step 5: Commit lifecycle and migration**

```bash
git add deepseek-api/agent.py deepseek-api/tests/test_agent.py
git commit -m "feat: persist active user profiles"
```

### Task 4: Strict profile HTTP API

**Files:**
- Modify: `deepseek-api/web.py`
- Modify: `deepseek-api/tests/test_web.py`

- [ ] **Step 1: Write failing HTTP tests**

Add tests for `GET /api/profiles` (`200`), `POST /api/profiles` (`201`), `PUT /api/profiles/profile-2` (`200`), and `POST /api/profiles/profile-2/activate` (`200`, no body). Include malformed/extra fields, duplicate names, a non-empty activation body, unknown ids, foreign origin, and a patched persistence failure; assert the specified `400`/`404`/`403`/`500` responses and unchanged state.

- [ ] **Step 2: Run the focused API tests and observe RED**

Run: `python3 -m unittest tests.test_web.DeepSeekWebTests.test_profile_routes_create_edit_and_activate -v`

Expected: FAIL with 404 because the routes do not exist.

- [ ] **Step 3: Implement handlers using existing safety patterns**

Route `GET /api/profiles`, `POST /api/profiles`, `PUT /api/profiles/<id>`, and `POST /api/profiles/<id>/activate` before generic agent route matching. Reuse same-origin checks, bounded body parsing, and `PersistenceError` conversion. Ensure activation accepts no content and that no malformed request reaches the registry.

- [ ] **Step 4: Run web API tests and observe GREEN**

Run: `python3 -m unittest tests.test_web -v`

Expected: PASS, including existing agent, memory, and context endpoints.

- [ ] **Step 5: Commit the API**

```bash
git add deepseek-api/web.py deepseek-api/tests/test_web.py
git commit -m "feat: expose profile management API"
```

### Task 5: Picker and modal UI

**Files:**
- Modify: `deepseek-api/web.py`
- Create: `deepseek-api/static/profile-controls.js`
- Create: `deepseek-api/static/profile-state.js`
- Modify: `deepseek-api/static/index.html`
- Modify: `deepseek-api/tests/test_client_state.py`
- Modify: `deepseek-api/tests/test_web.py`

- [ ] **Step 1: Write failing page-contract tests**

Assert the server serves `/static/profile-state.js` and `/static/profile-controls.js` as JavaScript `200`; the page loads both modules, has a `profile-picker`, a `profile-create-button`, edit/create dialogs with all four form fields, and the focused script includes `GET /api/profiles`, `POST /api/profiles`, `PUT`, `/activate`, explicit save/cancel actions, and a `flushProfileSave()` path before the composer request.

In the existing Node-backed client-state test style, add profile-state cases that verify a failed activation preserves `menuOpen`, keeps `editorProfileId` empty and makes `canSend` false; a successful activation sets the selected id and opens its editor; and a failed create/edit save keeps `canSend` false until a later successful state transition.

- [ ] **Step 2: Run the focused page test and observe RED**

Run: `python3 -m unittest tests.test_web.DeepSeekWebTests.test_page_exposes_profile_picker_and_dialogs tests.test_client_state.ProfileClientStateTests -v`

Expected: FAIL because no controls or script exist.

- [ ] **Step 3: Implement a focused global controller**

Create `ProfileClientState` in `profile-state.js` with pure `activationSucceeded`, `activationFailed`, `saveFailed`, `saveSucceeded`, and `canSend` functions; its only input/output is serializable browser state. Load it before `ProfileControls`.

Create `ProfileControls.mount(deps)`, patterned after existing memory/context controls. It loads the collection once, renders the active profile and menu, persists activation before opening the edit dialog, creates a profile through the `+` action, and returns `flushProfileSave()` plus `refresh()`. It must derive visible menu/editor/send states through `ProfileClientState`, so the Node test covers the required failure behavior without a DOM emulator.

Add named static file serving branches in `web.py` for both new scripts. Update the inline app state with global profile busy/error state. Mount the controller near the top of the metadata panel, call its flush function before switching/deleting/creating agents and before sending a message, disable actions while mutations are in flight, and surface errors without changing the menu or sending the chat request. Keep dialogs accessible (`role="dialog"`, labels, close/cancel controls) and do not add deletion.

- [ ] **Step 4: Run static and HTTP UI tests and observe GREEN**

Run: `python3 -m unittest tests.test_client_state tests.test_memory_ui tests.test_web -v`

Expected: PASS; the page contract confirms profile controls coexist with memory controls and agent settings.

- [ ] **Step 5: Commit UI changes**

```bash
git add deepseek-api/web.py deepseek-api/static/profile-state.js deepseek-api/static/profile-controls.js deepseek-api/static/index.html deepseek-api/tests/test_client_state.py deepseek-api/tests/test_web.py
git commit -m "feat: add profile picker and dialogs"
```

### Task 6: Documentation and end-to-end verification

**Files:**
- Modify: `deepseek-api/README.md`

- [ ] **Step 1: Document Day 12 behavior**

Add a «День 12 — персонализация ассистента» section explaining the active global profile, picker/dialog workflow, automatic prompt application, persistence, JSON-mode behavior (provider option only), and a two-profile manual check.

- [ ] **Step 2: Run the complete test suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: all tests pass. Run outside the sandbox when needed because web tests bind a temporary `127.0.0.1` HTTP server.

- [ ] **Step 3: Verify requirements against the specification**

Inspect the diff and confirm: profiles are persisted globally; style, format, and constraints are rendered into every user-facing agent request; profile changes apply across agents; the picker and both dialogs exist; JSON text is not appended; and the README describes validation.

- [ ] **Step 4: Commit documentation and final verification state**

```bash
git add deepseek-api/README.md
git commit -m "docs: describe day 12 personalization"
git status --short
```
