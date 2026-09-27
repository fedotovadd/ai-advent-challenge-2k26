# MCP Chat Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show real MCP tool-call progress inside the chat while preserving the existing JSON response for requests that do not ask for a tool stream.

**Architecture:** Add an optional callback through `AgentRegistry.respond`, `Agent.respond`, and `run_tools`. The HTTP endpoint enables NDJSON only when the browser requests MCP progress; it emits sequenced start/completed/failed events and a terminal snapshot. The existing chat client consumes those events and renders a compact progress card in the active thread.

**Tech Stack:** Python `http.server`, existing MCP registry/runtime, NDJSON, browser JavaScript and CSS.

---

### Task 1: Emit truthful MCP lifecycle events

**Files:**
- Modify: `deepseek-api/tool_runtime.py`
- Modify: `deepseek-api/agent.py`

- [x] Add optional `on_tool_event` arguments from `AgentRegistry.respond` through `Agent.respond` to `run_tools`.
- [x] Emit `mcp_tool_started` immediately before `registry.call`, after the tool name is resolved and flow inputs are accepted.
- [x] Emit `mcp_tool_completed` or `mcp_tool_failed` after the call; include the resolved server URL, tool name, bounded arguments, status, and duration.
- [x] Isolate callback failures so a closed browser connection does not change tool execution or chat persistence.

### Task 2: Stream MCP-enabled chat responses

**Files:**
- Create: `deepseek-api/mcp_chat_http.py`
- Modify: `deepseek-api/web.py`

- [x] When the browser includes `streamMcp: true`, send NDJSON headers and create a run identity with monotonically increasing event sequence numbers.
- [x] Forward lifecycle events, then send one terminal `mcp_chat_completed` or `mcp_chat_failed` event containing the final agent snapshot and response metadata.
- [x] Preserve current JSON behavior when `streamMcp` is absent or false.
- [x] Map existing chat/provider/persistence errors to a terminal stream event without leaking exception internals.

### Task 3: Render MCP events in the chat

**Files:**
- Modify: `deepseek-api/static/index.html`
- Create: `deepseek-api/static/mcp-progress.js`

- [x] Parse and validate MCP NDJSON events without changing the research-stream parser.
- [x] Render/upsert a compact in-chat status card as each MCP call starts and completes/fails.
- [x] Show friendly server/tool labels, ordered calls, and statuses; render external values as text.
- [x] Request the stream only when connected tools are available and the active model/format can use them.
- [x] Keep progress visible for the current page session after receiving the final agent snapshot.

### Task 4: Documentation and static verification

**Files:**
- Modify: `deepseek-api/README.md`

- [x] Document the optional `streamMcp` flow and the meaning of the in-chat statuses.
- [x] Run syntax compilation and `git diff --check`; do not add or run tests unless the user asks.
