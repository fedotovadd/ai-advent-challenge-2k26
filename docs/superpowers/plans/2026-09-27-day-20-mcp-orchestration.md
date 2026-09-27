# Day 20 MCP Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the chat agent orchestrate bounded multi-step requests across registered MCP servers and validate the resulting calls before reporting side effects.

**Architecture:** Preserve MCP `outputSchema` during discovery and use it to validate successful structured results. Add narrow result contracts for the demo's file-save and scheduler tools, and make the existing model-driven tool loop enforce a six-call and six-round budget before parsing or routing each accepted tool call. Keep server selection and tool order with the model; `McpRegistry` remains responsible for routing.

**Tech Stack:** Python 3.14, MCP Python SDK 1.x, Pydantic, jsonschema, existing unittest suite.

**Spec:** `docs/superpowers/specs/2026-09-27-day-20-mcp-orchestration-design.md`

---

## Files and responsibilities

- `deepseek-api/mcp_client.py`: discover both input and output schemas; retain existing call result bounds.
- `deepseek-api/mcp_registry.py`: retain the advertised schemas per connected tool and validate structured results where an output schema is present; validate the explicit demo result contracts.
- `deepseek-api/tool_runtime.py`: use the six-call/six-tool-round limits; consume call budget before parsing/routing; reject over-budget batch entries locally; preserve each result in `metadata.toolCalls`.
- `deepseek-api/README.md`: explain setup and the repeatable three-server scenario.
- `docs/day-20/live-results.json`, `docs/day-20/orchestration-results.md`: record only actual live-run evidence, when the local servers and external GitHub are available.

Implementation constraint: current task instructions prohibit adding or running tests unless the user asks for verification. Do not add or execute tests in this pass. Use syntax compilation and diff inspection as static checks, and clearly report that the automated suite was not run.

## Task 1: Preserve tool output schemas during discovery

**Files:**
- Modify: `deepseek-api/mcp_client.py`
- Modify: `deepseek-api/mcp_registry.py`

- [x] Include the MCP `outputSchema` in each discovered tool when present, alongside `inputSchema`.
- [x] Validate discovered output schemas with the same bounded local-schema rules used for input schemas; do not fetch remote references.
- [x] Keep `outputSchema` out of the model-facing function definition, since provider tool definitions accept input parameters only.
- [x] Keep enough server/tool metadata in `McpRegistry` to validate a call result against the matching output schema after invocation.
- [x] For a successful result with a published output schema, require object `structuredContent` and validate it; convert missing or invalid structured output to a bounded `isError` result.
- [x] Preserve existing behavior for tools without an output schema, except the explicit scenario contracts in Task 2.
- [x] Review the diff for compatibility with existing URL allowlists, discovery return contracts, and old persisted server records.

## Task 2: Validate side-effect results needed by the scenario

**Files:**
- Modify: `deepseek-api/mcp_registry.py`
- Modify: `deepseek-api/tool_runtime.py`

- [x] Add focused validators for local Pipeline `search` and `summarize` results so required structured data exists even when the server does not publish `outputSchema`.
- [x] Add focused validators for `saveToFile`: nonempty string `path`, positive integer `bytes_written` (reject bool), and lowercase 64-character SHA-256.
- [x] Compare the saved byte count and SHA-256 with the exact UTF-8 bytes of the current turn's validated summary.
- [x] Add a focused validator for `schedule_repository`: require `job.id`, matching `owner`/`repo`, active status, null interval, and timezone-aware `next_run_at`.
- [x] Validate `get_schedule_summary` against its requested `job_id` and nonnegative counters. `tool_runtime` keeps turn-local job IDs and repo pairs; when a result confirms a schedule created in this turn, require matching repository, null interval, and exactly zero success/error counts before the delayed first run. Preserve existing summaries with periods and history.
- [x] Apply these checks only to the matching tools on the configured local Pipeline and Scheduler MCP endpoints, so a public server with a colliding tool name is not treated as the trusted local contract.
- [x] Convert contract violations to bounded MCP errors so the model cannot treat malformed results as proof of a side effect.

## Task 3: Bound model-driven orchestration

**Files:**
- Modify: `deepseek-api/tool_runtime.py`

- [x] Set the accepted tool-call budget and tool-enabled model-round budget to six.
- [x] Count each accepted `tool_call` before parsing its arguments, resolving its name, or validating its schema; malformed JSON, unknown tools, and disconnected-server calls consume budget.
- [x] For call batches that exceed the remaining budget, return local errors for excess entries without dispatching them to an MCP server.
- [x] Keep the final model request at `tool_choice='none'` once the tool-call or round limit is reached.
- [x] Preserve ordering, matching tool-call IDs, safe errors, and actual per-call server URL/name in the existing trace.
- [x] Ensure the runtime never dispatches more than six MCP calls during one user turn.
- [x] Keep turn-local search results, selected repo, summary, and newly created job mappings to validate exact data handoffs without persisting context across turns.
- [x] After a failed `schedule_repository`, reject further Scheduler tools in that chat turn; a new user turn may inspect existing schedules.

## Task 4: Document and run the scenario

**Files:**
- Modify: `deepseek-api/README.md`
- Create after an actual run: `docs/day-20/live-results.json`
- Create after an actual run: `docs/day-20/orchestration-results.md`

- [x] Document startup commands for `pipeline_mcp_server.py`, `github_mcp_server.py`, and `scheduler_mcp_server.py`, and connect their existing loopback MCP URLs in the MCP tab.
- [x] Document the natural-language scenario from the spec and the expected six-tool ordering across three servers.
- [x] Run `.venv/bin/python -m py_compile mcp_client.py mcp_registry.py tool_runtime.py` from `deepseek-api/` and inspect the resulting diff; do not run unittest in this pass.
- [ ] Execute the scenario in chat and save the actual trace/results to `docs/day-20/`. Not run because the current task instruction prohibits running verification scenarios unless the user asks for verification; do not fabricate evidence.
- [x] Check `git diff --check` and final working-tree status.

## Completion criteria

- At least two connected MCP servers appear in the same model tool snapshot and each selected function routes to its owning server.
- The six-step demonstration can execute through the normal chat tool loop under the six-call budget.
- Invalid/missing output from `saveToFile` or the scheduler cannot be presented as a successful side effect.
- Calls beyond the budget and malformed/unknown calls cannot cause unbounded MCP dispatch.
- `/research` and existing MCP connection behavior remain unchanged by inspection.
- The code compiles. Automated tests are explicitly reported as not run under the current instruction constraint.
