# Temperature Setting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a chat user select a temperature in the web interface and send it to DeepSeek with every message request.

**Architecture:** Keep the selected temperature in browser state through the range input's current value. The browser sends it alongside `text`; the HTTP handler normalizes and validates it before building the provider payload. The existing metadata response exposes the exact payload unchanged, including `temperature`.

**Tech Stack:** Python 3 standard-library HTTP server, `unittest`, embedded browser-native HTML/CSS/JavaScript, OpenAI Python client.

---

### Task 1: Validate and forward temperature on the HTTP endpoint

**Files:**
- Modify: `deepseek-api/tests/test_main.py`
- Modify: `deepseek-api/main.py:153-179`

- [ ] **Step 1: Write the failing endpoint tests**

Add a test covering the three required values. For each of `0`, `0.7`, and `1.2`, send a fresh message request with both fields and assert a 200 response, the recorded model payload and `body["metadata"]["payload"]` include the exact value:

```python
status, body, _ = self.json_request(
    "POST", "/api/sessions/session-1/messages",
    {"text": "Привет", "temperature": 0.7},
)

self.assertEqual(status, 200)
self.assertEqual(self.payloads[-1]["temperature"], 0.7)
self.assertEqual(body["metadata"]["payload"]["temperature"], 0.7)
```

Update the existing `test_message_returns_answer_and_exact_metadata` expectations for both the recorded provider payload and `metadata["payload"]` to include the backward-compatible default, `"temperature": 1`.

Add a backward-compatibility test sending only `text` and expecting `temperature == 1`. Add subtests for `"0.7"`, `None`, `True`, `-0.1`, `2.1`, `float("inf")`, and `float("nan")`; each must return 400, contain `{"error": "Поле temperature должно быть числом от 0 до 2."}`, and make no call to `ask_model`.

- [ ] **Step 2: Run the focused tests and confirm the expected failure**

Run:

```bash
cd deepseek-api && python3 -m unittest tests.test_main.DeepSeekWebTests.test_message_forwards_requested_temperature -v
```

Expected: FAIL because the handler neither reads nor adds `temperature` to the payload.

- [ ] **Step 3: Implement the minimum server-side contract**

In `main.py`, define `DEFAULT_TEMPERATURE = 1`. In `ChatRequestHandler._handle_message`, read `data.get("temperature", DEFAULT_TEMPERATURE)` after parsing JSON. Reject booleans and non-`int`/`float` values, plus non-finite numbers and numbers outside the inclusive `0`–`2` range, with the error message used by the test. Preserve `0` as valid. Build the payload as:

```python
payload = {
    "model": MODEL,
    "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *session["messages"]],
    "temperature": temperature,
}
```

Update `ask_deepseek` to call the provider with `temperature=payload["temperature"]` in addition to the existing model and messages. Do not change session storage or provider-error handling.

- [ ] **Step 4: Run the focused tests and confirm they pass**

Run:

```bash
cd deepseek-api && python3 -m unittest tests.test_main.DeepSeekWebTests.test_message_forwards_requested_temperature tests.test_main.DeepSeekWebTests.test_message_without_temperature_uses_default tests.test_main.DeepSeekWebTests.test_invalid_temperature_is_rejected -v
```

Expected: PASS for all temperature contract tests.

- [ ] **Step 5: Run the complete backend test suite**

Run:

```bash
cd deepseek-api && python3 -m unittest discover -v
```

Expected: PASS with no failures.

- [ ] **Step 6: Commit the server contract**

```bash
git add deepseek-api/main.py deepseek-api/tests/test_main.py
git commit -m "feat: send temperature to DeepSeek"
```

### Task 2: Add the response-settings range control to the web page

**Files:**
- Modify: `deepseek-api/tests/test_main.py`
- Modify: `deepseek-api/main.py:20-59`

- [ ] **Step 1: Write the failing page-contract test**

Extend `test_page_contains_chat_controls` to assert that the served page contains a response-settings heading, a range input and the configured default:

```python
self.assertIn("Настройки ответа", body)
self.assertIn('id="temperature"', body)
self.assertIn('type="range"', body)
self.assertIn('min="0"', body)
self.assertIn('max="2"', body)
self.assertIn('step="0.1"', body)
self.assertIn('value="1"', body)
```

Also assert the page script sends the temperature field by checking for `temperature:Number(elements.temperature.value)`.

- [ ] **Step 2: Run the focused test and confirm the expected failure**

Run:

```bash
cd deepseek-api && python3 -m unittest tests.test_main.DeepSeekWebTests.test_page_contains_chat_controls -v
```

Expected: FAIL because the settings control is absent.

- [ ] **Step 3: Implement the smallest accessible interface change**

Add a «Настройки ответа» card at the top of the existing right-hand `.metadata` panel. Its labelled `<input id="temperature" type="range" min="0" max="2" step="0.1" value="1">` should be accompanied by an output element initially showing `1`. Add focused CSS so the label, current value and slider fit the existing card style and narrow-screen layout.

Extend the `elements` object with the range input and output. On its `input` event, set the output text to the slider's value. In the composer submit request, send `JSON.stringify({text, temperature:Number(elements.temperature.value)})`. Leave the selected value untouched when sending, changing sessions, or receiving API errors.

- [ ] **Step 4: Run the focused page-contract test and confirm it passes**

Run:

```bash
cd deepseek-api && python3 -m unittest tests.test_main.DeepSeekWebTests.test_page_contains_chat_controls -v
```

Expected: PASS.

- [ ] **Step 5: Run all automated tests**

Run:

```bash
cd deepseek-api && python3 -m unittest discover -v
```

Expected: PASS with no failures.

- [ ] **Step 6: Manually verify the interaction**

Run the local server with an API key configured, open `http://127.0.0.1:8000`, move the slider successively to `0`, `0.7`, and `1.2`, and send a message after each choice. Confirm that the visible value follows the slider and «Отправлено в API» shows the same number in `temperature` for each request.

- [ ] **Step 7: Commit the interface**

```bash
git add deepseek-api/main.py deepseek-api/tests/test_main.py
git commit -m "feat: add response temperature slider"
```
