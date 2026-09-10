# Token Metrics and Context Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every agent an isolated, persistent-in-memory view of token usage, cost, context capacity, and overflow behavior, including an opt-in provider-error demonstration in the chat.

**Architecture:** `agent.py` will own token estimation, context preflight, successful call records, totals, and the temporary overflow probe transaction. `web.py` will validate new per-agent settings and expose provider error text only for the opted-in probe. `static/index.html` will render the metrics panel, per-agent controls, chart/table, and chat feedback from each agent snapshot.

**Tech Stack:** Python 3 standard library, `http.server`, existing OpenAI-compatible provider adapter, vanilla HTML/CSS/JavaScript, `unittest`.

---

### Task 1: Model limits and pure metric calculations

**Files:**
- Modify: `agent.py:10-88`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write failing unit tests for the pure helpers**

Add tests that establish the public data shape and arithmetic. Fix capabilities from the providers' documentation: `deepseek-v4-flash` and `deepseek-v4-pro` use `contextLimit=1_000_000` and `maxOutputTokens=384_000`; `glm-4.7-flash` uses `contextLimit=200_000` and `maxOutputTokens=131_072`. All three use the conservative `safeOutputTokens=4_096` when the user did not set `maxTokens`. The smaller training limit is the reproducible overflow control.

```python
def test_estimate_tokens_counts_text_and_message_structure(self):
    self.assertGreater(agent.estimate_text_tokens("один два"), 0)
    request = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "Будь кратким"},
            {"role": "user", "content": "Привет"},
        ],
        "temperature": 1,
        "max_tokens": 300,
    }
    self.assertGreater(
        agent.estimate_payload_tokens(request),
        agent.estimate_text_tokens("Привет"),
    )
    self.assertGreater(
        agent.estimate_payload_tokens(request),
        agent.estimate_payload_tokens({**request, "messages": request["messages"][1:]}),
    )

def test_request_usage_uses_actual_usage_or_explicit_estimates(self):
    actual = agent.request_usage_and_cost("deepseek-v4-flash", {"prompt_tokens": 10, "completion_tokens": 5})
    estimated = agent.request_usage_and_cost("deepseek-v4-flash", None, 12, 4)
    self.assertEqual(actual[0]["source"], "actual")
    self.assertEqual(estimated[0]["source"], "estimated")

def test_partial_usage_uses_only_estimates_without_mixing_sources(self):
    usage, cost = agent.request_usage_and_cost(
        "deepseek-v4-flash", {"prompt_tokens": 10}, 12, 4
    )
    self.assertEqual(usage, {"promptTokens": 12, "completionTokens": 4,
                             "totalTokens": 16, "source": "estimated"})

def test_model_capabilities_keep_provider_context_and_output_maxima(self):
    self.assertEqual(agent.MODEL_CAPABILITIES["deepseek-v4-pro"]["maxOutputTokens"], 384_000)
    self.assertEqual(agent.MODEL_CAPABILITIES["glm-4.7-flash"]["contextLimit"], 200_000)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `python3 -m unittest tests.test_agent.AgentTests.test_estimate_tokens_counts_text_and_message_structure tests.test_agent.AgentTests.test_request_usage_uses_actual_usage_or_explicit_estimates tests.test_agent.AgentTests.test_partial_usage_uses_only_estimates_without_mixing_sources -v`

Expected: FAIL because the estimators and new `source` field do not exist.

- [ ] **Step 3: Add model context configuration and minimal helpers**

In `agent.py`, add `MODEL_CAPABILITIES` with the fixed per-model `contextLimit`, `maxOutputTokens`, and `safeOutputTokens` values, a deterministic local text/message/payload estimator, and a helper that creates normalized usage data. Validate a requested `settings.maxTokens` against the selected model's `maxOutputTokens`; use `safeOutputTokens` only when it is absent. The payload estimator must receive the whole request object, including the system message, all chat messages, model/options and a fixed message-structure overhead:

```python
{"promptTokens": prompt, "completionTokens": completion,
 "totalTokens": total, "source": "actual" | "estimated"}
```

For a paid model calculate USD using the existing per-million input/output rates; retain numerical zero plus `kind: "free"` for free models. Do not round stored USD values.

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `python3 -m unittest tests.test_agent.AgentTests.test_estimate_tokens_counts_text_and_message_structure tests.test_agent.AgentTests.test_request_usage_uses_actual_usage_or_explicit_estimates tests.test_agent.AgentTests.test_partial_usage_uses_only_estimates_without_mixing_sources -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: add token metric calculations"
```

### Task 2: Per-agent metric state, successful records, and persistence

**Files:**
- Modify: `agent.py:90-240`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write failing tests for a successful metric record and isolation**

Add a response with known usage, then assert the snapshot has one record and derived totals. The record contract is:

```python
{
  "messageNumber": 1,
  "messageTokens": 3,
  "historyBeforeTokens": 0,
  "payloadEstimatedTokens": 17,
  "promptTokens": 100, "completionTokens": 50, "totalTokens": 150,
  "source": "actual", "historyAfterTokens": 24,
  "contextLimit": 1_000_000, "contextPercent": 0.0017,
  "cost": {"kind": "paid", "usd": ...},
  "cumulativePromptTokens": 100, "cumulativeCompletionTokens": 50,
  "cumulativeTotalTokens": 150, "cumulativeUsd": ...,
}
```

Create a second agent and assert it starts with no records and zero totals:

```python
self.assertEqual(snapshot["metrics"]["calls"][0]["promptTokens"], 100)
self.assertEqual(snapshot["metrics"]["totals"]["totalTokens"], 150)
self.assertEqual(second.snapshot()["metrics"]["calls"], [])
```

Also test no-usage and partial-usage fallback, a free call, and registry save/reload of the metric state. Update old full-snapshot/settings assertions and saved-state fixtures to the expanded schema as part of this red step.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python3 -m unittest tests.test_agent.AgentTests.test_successful_response_records_session_metrics tests.test_agent.AgentRegistryTests.test_registry_restores_metrics_per_agent -v`

Expected: FAIL because snapshots do not contain `metrics`.

- [ ] **Step 3: Extend the agent snapshot and persistence schema**

Add one explicit `metrics` object to `Agent`: `calls`, derived `totals`, `lastMessage`, and `lastContextAttempt`. Only a successful model response appends a call record, using the exact contract above. `lastMessage` is a copy of the newest successful record. `lastContextAttempt` is `{payloadEstimatedTokens, requestedOutputTokens, contextLimit, contextPercent, reason, mode}` and does not affect totals.

Increment `STATE_VERSION` to 2 and add a v1-to-v2 migration before strict validation: supply `trainingContextLimit=None`, `sendOnOverflow=False`, and an empty default metrics object to every valid v1 agent. Add a test that loading a valid v1 file preserves its messages/settings/metadata and gains these defaults. Update agent construction, snapshot validation, registry restore and default creation to accept the new object. Preserve all existing snapshot copies and lock behavior. Settings gain nullable `trainingContextLimit` and boolean `sendOnOverflow` with strict validation and defaults.

- [ ] **Step 4: Run focused tests and full non-HTTP agent tests**

Run: `python3 -m unittest tests.test_agent -v`

Expected: PASS, including existing persistence/isolation tests.

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: persist per-agent token metrics"
```

### Task 3: Context preflight and atomic overflow-probe behavior

**Files:**
- Modify: `agent.py:115-165`
- Modify: `web.py:190-240`
- Test: `tests/test_agent.py`
- Test: `tests/test_web.py`

- [ ] **Step 1: Write failing agent and HTTP tests**

Cover all three branches, including provider exception shapes with an SDK-like `.body` and a Bearer/API-key-shaped value that must be masked:

```python
def test_overflow_without_demo_call_does_not_call_provider_or_change_messages(self):
    # trainingContextLimit forces input + reserved answer over the limit
    with self.assertRaises(agent.ContextOverflowError):
        self.agent.respond("слишком длинное сообщение")
    self.assertEqual(self.calls, [])
    self.assertEqual(self.agent.snapshot()["messages"], [])

def test_overflow_probe_exposes_provider_error_without_committing_message(self):
    # sendOnOverflow=True; fake provider raises an error carrying `error.message`
    ...

def test_overflow_probe_success_commits_user_and_answer_once(self):
    ...
```

At the HTTP boundary, assert a local overflow returns a clear 4xx response with the updated snapshot; assert an opted-in provider failure returns the extracted provider message and leaves its message history unchanged.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python3 -m unittest tests.test_agent.AgentTests.test_overflow_without_demo_call_does_not_call_provider_or_change_messages tests.test_web.DeepSeekWebTests.test_overflow_probe_returns_provider_error_for_chat -v`

Expected: FAIL because context preflight and the probe response do not exist.

- [ ] **Step 3: Implement preflight before mutating history**

Build the candidate payload from a copy of current messages. Select active limit as `trainingContextLimit` when present, otherwise the model context limit. Compute display percent from input-only estimate and test capacity with input plus `maxTokens` or the safe default. At 80–99% include a warning in metrics; when capacity fails, always populate `lastContextAttempt` without changing successful totals.

When `sendOnOverflow` is false, raise a dedicated `ContextOverflowError`. When true, call the provider using the candidate payload and only append the user/assistant pair after a valid answer. On failure, re-raise a dedicated probe error containing the provider-derived error string. Extract nested `body["error"]["message"]`, then `body["message"]`, then exception `message`; otherwise use HTTP status plus body. Before exposure, replace `Bearer <non-whitespace>`, values under JSON keys containing `key`, `token`, `secret`, or `authorization`, and configured API-key values with `[скрыто]`. Test all extraction/fallback and redaction paths.

- [ ] **Step 4: Map the dedicated errors in the HTTP handler**

Keep ordinary provider errors on the existing generic path. Add only overflow branches that return the refreshed agent snapshot and `metadata`, with the local Russian error or the provider-error text requested by the user. Keep the error JSON safe for malformed/non-JSON upstream failures.

- [ ] **Step 5: Run focused and full server tests**

Run: `python3 -m unittest tests.test_agent tests.test_web -v`

Expected: PASS. Run with local-port permission if the restricted environment cannot bind `127.0.0.1`.

- [ ] **Step 6: Commit**

```bash
git add agent.py web.py tests/test_agent.py tests/test_web.py
git commit -m "feat: guard agent context overflow"
```

### Task 4: Settings validation and metrics UI

**Files:**
- Modify: `web.py:267-320`
- Modify: `static/index.html:8-98`
- Test: `tests/test_web.py`

- [ ] **Step 1: Write failing server/UI contract tests**

Add tests that settings accept valid `trainingContextLimit`/`sendOnOverflow`, reject incorrect values without replacement, and that the static page includes the token section, warning area, checkbox, chart/table headings, and the exact rendering path for a server-provided provider error. Update all existing complete settings/snapshot expectations to include the two default settings and empty metrics object.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python3 -m unittest tests.test_web.DeepSeekWebTests.test_settings_validate_context_demo_controls tests.test_web.DeepSeekWebTests.test_page_contains_token_metrics_panel -v`

Expected: FAIL because fields and UI identifiers are absent.

- [ ] **Step 3: Validate and persist the two controls through the existing settings route**

Extend the exact settings field set, checks and frontend `settingsValues()`/`renderSettings()` together. The checkbox must default to unchecked for every newly created or restored compatible agent; a training limit is a positive integer or null.

- [ ] **Step 4: Render agent-scoped metrics and chat feedback**

Extend the right panel with:

```text
Токены и стоимость
Реплик в диалоге / вход / выход / всего / стоимость / источник данных
Размер сообщения / история до и после / payload / ответ / контекст
Расход по сообщениям (SVG polyline) + таблица
```

Use `textContent`, not `innerHTML`, for provider failures. Keep a per-agent transient attempt in client state for an opted-in rejected probe: render its submitted user text and provider error together, but do not place either in `agent.messages`; clear it after the next successful response, another submission, agent deletion, or page reload. For locally blocked overflow, remove the optimistic user message when applying the server snapshot and show only the local context error. Render warning/error from `lastContextAttempt`, and ensure all metrics state comes from `activeAgent()` rather than a global aggregate. Add a client/UI test covering the transient probe pair.

- [ ] **Step 5: Run focused tests and full suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: PASS. Run with local-port permission if necessary.

- [ ] **Step 6: Commit**

```bash
git add web.py static/index.html tests/test_web.py
git commit -m "feat: show agent token metrics"
```

### Task 5: User documentation and final verification

**Files:**
- Modify: `README.md`
- Modify: `deepseek-api/README.md`
- Test: `tests/test_agent.py`
- Test: `tests/test_web.py`

- [ ] **Step 1: Write documentation tests only if existing documentation contracts require them**

Do not create brittle prose tests. Use the existing functional suite as the behavior contract.

- [ ] **Step 2: Document Day 8 and the three demonstrations**

Add the Day 8 entry to the root task list and a concise project README section covering metrics source labels, pricing, training limit, the opt-in provider-error check box, and short/long/overflow demonstration instructions.

- [ ] **Step 3: Run final verification**

Run: `python3 -m unittest discover -s tests -v`

Expected: PASS with all pre-existing and new tests.

- [ ] **Step 4: Manually inspect the interface**

Start `python3 main.py`, open the local page, and verify one short dialogue, an 80% warning and a local block using a training limit. Verify the opted-in provider-error chat rendering with the deterministic fake provider test; manually observe it against a real provider only if that provider actually rejects the payload, because the training limit does not constrain the provider itself. Stop the local server after inspection.

- [ ] **Step 5: Commit**

```bash
git add README.md deepseek-api/README.md
git commit -m "docs: explain day 08 token metrics"
```
