# Category Memory Commands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow chat commands to write to and clear every named category of working and long-term memory.

**Architecture:** The command parser returns one explicit action for each target layer. `Agent` mutates only the addressed substructure, and the existing registry persists the resulting snapshot. The memory panel remains read-only.

**Tech Stack:** Python 3, unittest, built-in HTTP server, vanilla JavaScript.

---

### Task 1: Parse and apply category commands

**Files:**
- Modify: `deepseek-api/memory_layers.py`, `deepseek-api/agent.py`
- Test: `deepseek-api/tests/test_memory_layers.py`, `deepseek-api/tests/test_agent.py`

- [ ] Add failing tests for `/working-data key: value`, `/profile key: value`, `/decision text`, `/knowledge key: value`, and four category-specific clear actions.
- [ ] Run the focused tests and confirm the parser rejects the unknown actions.
- [ ] Parse `key: value` pairs without allowing empty sides. Mutate only the named object/list; clear only the named target. Retire ambiguous `/long`.
- [ ] Re-run focused tests.

### Task 2: HTTP integration and documentation

**Files:**
- Modify: `deepseek-api/tests/test_web.py`, `docs/day-11/memory-layers-results.md`

- [ ] Add a failing message-route test proving commands make no model call, persist, and do not alter unrelated categories.
- [ ] Run it to verify red.
- [ ] Use the existing command registry path; update the Day 11 usage reference.
- [ ] Run focused and full tests, check JavaScript syntax, then commit.
