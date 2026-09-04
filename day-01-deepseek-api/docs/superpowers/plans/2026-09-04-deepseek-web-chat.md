# DeepSeek Web Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local browser chat for DeepSeek with sessions and transparent API-request metadata.

**Architecture:** `main.py` becomes a dependency-injected HTTP-server factory backed by an in-memory session store. It serves one self-contained HTML page and JSON endpoints; the browser uses fetch to create sessions and send messages. A small isolated function creates the DeepSeek request and returns only safe response data to the HTTP layer.

**Tech Stack:** Python 3 standard-library HTTP server, `openai` Python package, browser-native HTML/CSS/JavaScript, `unittest`.

---

### Task 1: Establish the HTTP server contract in tests

**Files:**
- Modify: `day-01-deepseek-api/tests/test_main.py`
- Modify: `day-01-deepseek-api/main.py`

- [ ] **Step 1: Write the failing tests for page and session lifecycle**

Replace the CLI tests with an HTTP test helper that starts `main.create_server(host="127.0.0.1", port=0, ask_model=fake)` in a daemon thread and makes requests with `http.client`. Add tests asserting:

```python
status, body, headers = request("GET", "/")
self.assertEqual(status, 200)
self.assertIn("text/html", headers["Content-Type"])
self.assertIn("DeepSeek", body)

status, body, _ = request("GET", "/api/sessions")
self.assertEqual(status, 200)
self.assertEqual(json.loads(body)["sessions"], [
    {"id": "session-1", "title": "Новый чат", "messages": []}
])
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python3 -m unittest tests.test_main -v`

Expected: FAIL because `create_server` does not exist.

- [ ] **Step 3: Implement only the server factory, GET routes and in-memory session store**

Add `create_server(host, port, ask_model=ask_deepseek)` returning a `ThreadingHTTPServer`, a request handler that returns the static page at `/`, JSON at `GET /api/sessions`, and JSON 404 elsewhere. Initialise `session-1` in a small `SessionStore`. Protect all sequential id allocation, read-modify-write history updates and snapshots with a `threading.Lock` so concurrent HTTP requests cannot race.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python3 -m unittest tests.test_main -v`

Expected: PASS for the page and session lifecycle tests.

- [ ] **Step 5: Commit**

```bash
git add day-01-deepseek-api/main.py day-01-deepseek-api/tests/test_main.py
git commit -m "feat: serve DeepSeek chat sessions"
```

### Task 2: Add test-driven message delivery and error handling

**Files:**
- Modify: `day-01-deepseek-api/tests/test_main.py`
- Modify: `day-01-deepseek-api/main.py`

- [ ] **Step 1: Write failing success-path and validation tests**

Set `DEEPSEEK_API_KEY` in successful fake-backed tests. Use a fake `ask_model(payload)` recording the exact payload and returning a fixed answer. Assert `POST /api/sessions/session-1/messages` with `{"text": "Привет"}` returns `200`, includes the documented success metadata, appends user and assistant messages, changes the title to `Привет`, and calls the fake with:

```python
[
    {"role": "system", "content": main.SYSTEM_PROMPT},
    {"role": "user", "content": "Привет"},
]
```

Send a second message and assert its payload contains, in order, system prompt, prior user message, prior assistant message and the new user message. Add a test that a first message longer than 40 characters creates a title truncated to 40 characters. Add separate tests for empty text, missing `text`, non-string `text` and malformed JSON (`400`), unknown session (`404`), and `POST /api/sessions` (`201`, `session-2`).

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python3 -m unittest tests.test_main -v`

Expected: FAIL because POST routes are missing.

- [ ] **Step 3: Implement minimal POST routes and payload metadata**

Decode and validate JSON once, trim outer whitespace from accepted text before storing, deriving the title or forming the payload, create sessions sequentially, store user messages, build the exact `{model, messages}` payload with `SYSTEM_PROMPT` first, call `ask_model`, store the assistant response, and return the session plus safe metadata. Return defined 400 and 404 JSON error bodies. Add a test that input with outer whitespace is stored and sent trimmed.

- [ ] **Step 4: Write failing tests for provider failure and request origin**

Add a fake that raises `RuntimeError` and a fake that returns an empty string; assert each produces `502` with saved session, safe error text, full non-secret payload and error status. Add a test without `DEEPSEEK_API_KEY` asserting `503`. Add foreign-Origin POST tests for both session creation and message sending that assert `403` and no model call, plus a message POST with an exact matching `Origin: http://127.0.0.1:<temporary-port>` that succeeds.

- [ ] **Step 5: Run the focused tests to verify they fail**

Run: `python3 -m unittest tests.test_main -v`

Expected: FAIL because provider and origin cases are not handled.

- [ ] **Step 6: Implement error and origin safeguards**

Implement same-origin rejection by comparing an explicit `Origin` exactly with `http://{Host-header}` (including the actual temporary port). On missing key, provider exception or empty response, preserve the user message and return a safe error with session and metadata; never return exception internals or secrets. Implement `ask_deepseek(payload)`: read `DEEPSEEK_API_KEY`, create `OpenAI(api_key=key, base_url="https://api.deepseek.com")`, call `client.chat.completions.create(model="deepseek-v4-flash", messages=payload["messages"])`, and return nonempty extracted text. The injected `ask_model` replaces only that final provider call in tests; the handler owns the key check.

- [ ] **Step 7: Run the focused tests to verify they pass**

Run: `python3 -m unittest tests.test_main -v`

Expected: PASS for all endpoint and error tests.

- [ ] **Step 8: Commit**

```bash
git add day-01-deepseek-api/main.py day-01-deepseek-api/tests/test_main.py
git commit -m "feat: send DeepSeek messages from web chat"
```

### Task 3: Build the browser chat surface

**Files:**
- Modify: `day-01-deepseek-api/main.py`
- Test: `day-01-deepseek-api/tests/test_main.py`

- [ ] **Step 1: Write a failing static-page contract test**

Add test assertions that `/` contains session navigation, a message composer, a metadata heading and no raw API key placeholder.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest tests.test_main.DeepSeekWebTests.test_page_contains_chat_controls -v`

Expected: FAIL because the placeholder page has no chat controls.

- [ ] **Step 3: Implement one embedded page with accessible controls**

Define the HTML page constant in `main.py` with warm off-white and brick CSS. Use semantic buttons, textarea and status regions. Client JavaScript must load sessions, create a session, select sessions and send messages via fetch. For every response, including non-2xx responses, it must parse `{session, error, metadata}`, update the displayed session and metadata when supplied, show the safe error, and re-enable the composer in a `finally` path. Build all dynamic DOM nodes using `document.createElement` and `textContent`; never interpolate user/model data into `innerHTML`. Render user prompt, system prompt, payload and status from the last response in the right-hand panel. Add a narrow-screen media query which puts sessions above the chat and metadata below it.

- [ ] **Step 4: Run the static-page and full test suite**

Run: `python3 -m unittest discover -v`

Expected: PASS with no failures.

- [ ] **Step 5: Commit**

```bash
git add day-01-deepseek-api/main.py day-01-deepseek-api/tests/test_main.py
git commit -m "feat: add warm DeepSeek chat interface"
```

### Task 4: Document startup and final verification

**Files:**
- Modify: `day-01-deepseek-api/README.md`

- [ ] **Step 1: Update the README**

Replace the terminal-input steps with the local-server launch. Replace `main()` so it binds and serves `create_server("127.0.0.1", 8000)` and announces `http://127.0.0.1:8000` before serving:

```bash
python3 main.py
```

Then instruct the reader to open `http://127.0.0.1:8000`, send a message and inspect the metadata sidebar. Keep the key only in `DEEPSEEK_API_KEY`.

- [ ] **Step 2: Run all tests**

Run: `python3 -m unittest discover -v`

Expected: PASS with no failures.

- [ ] **Step 3: Smoke-test the server**

Run: `python3 main.py`

Expected: Server announces `http://127.0.0.1:8000`; loading that URL renders the chat UI. In the browser, create and select a second chat, send a message with a configured API key, confirm the response appears and the metadata panel shows the returned payload; temporarily simulate an API failure in the injected test server and confirm the safe error and metadata render while the composer becomes usable again.

- [ ] **Step 4: Commit**

```bash
git add day-01-deepseek-api/README.md
git commit -m "docs: explain DeepSeek web chat"
```
