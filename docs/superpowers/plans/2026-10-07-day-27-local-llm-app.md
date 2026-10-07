# Day 27 Local LLM Web Chat Implementation Plan

> **For agentic workers:** Execute the tasks in order, using test-driven development. Check each step against the Day 27 design.

**Goal:** Add `qwen2.5:1.5b` through Ollama's loopback OpenAI-compatible endpoint to the existing agent web chat.

**Architecture:** The provider adapter routes one exact model name to a fixed loopback URL and translates local failures. Agent settings and context limits include the local model, while the current HTTP and browser chat flows continue to own persistence and rendering.

**Tech Stack:** Python standard library HTTP server, OpenAI Python SDK, JavaScript, unittest.

---

### Task 1: Provider routing and errors

- [x] Add failing tests in `deepseek-api/tests/test_providers.py` for exact local URL, dummy key, no cloud key reads, one attempt, finite timeout, forwarded options, empty response, and SDK failure mapping.
- [x] Run `python -m unittest discover -s tests -p test_providers.py -v` and confirm relevant failures.
- [x] Implement the exact local route and error translation in `deepseek-api/providers.py`.
- [x] Run the provider tests again.

### Task 2: Agent behavior

- [x] Add failing tests in `deepseek-api/tests/test_local_model_agent.py` for settings persistence, isolation, 512 default output, 1024 maximum, context reservation, no MCP, and error history.
- [x] Run the agent tests and confirm relevant failures.
- [x] Update model declarations and local default output in `deepseek-api/agent.py`.
- [x] Run the agent tests again.

### Task 3: HTTP and browser

- [x] Add failing HTTP tests in `deepseek-api/tests/test_web.py` for local success, persistence, and the four error states.
- [x] Run the HTTP tests and confirm relevant failures.
- [x] Map local errors in `deepseek-api/web.py`, keeping the snapshot and unanswered user message.
- [x] Add the model selector and local cost note in `deepseek-api/static/index.html`, using text nodes for dynamic content.
- [x] Run HTTP and UI tests.

### Task 4: Instructions and live results

- [x] Update `deepseek-api/README.md` with Ollama startup, model check, browser steps, and no-key local mode.
- [x] Run focused and complete unit suites.
- [x] Check Ollama model and live two-message web chat without cloud keys, including failure and recovery where environment permits.
- [x] Write actual commands, outputs, settings, endpoint, and any limitations to `docs/day-27/results.md`.
