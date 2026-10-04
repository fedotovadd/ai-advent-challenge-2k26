# Day 22 First RAG Implementation Plan

> **For agentic workers:** Implement task by task with test-first cycles and verify each failure before production code.

**Goal:** Add CLI plain and RAG answers to book questions and compare ten controlled questions.

**Architecture:** `rag_agent.py` orchestrates existing provider and index APIs. A JSON file holds questions and expected sources. Evaluation writes machine results atomically and a Markdown report for manual grading.

**Tech Stack:** Python 3, unittest, SQLite index, local Ollama, Z.AI.

---

### Task 1: Control questions

**Files:** Create `docs/day-22/control_questions.json`.

- [x] Encode all ten approved questions, thesis points, and EPUB XHTML source names.
- [x] Validate count and fields by loading JSON and comparing to the approved spec.

### Task 2: Answer orchestration

**Files:** Create `deepseek-api/tests/test_rag_agent.py`; create `deepseek-api/rag_agent.py`.

- [x] Write tests for plain no-index access, shared provider configuration, RAG context and source order, validation, and service errors.
- [x] Run tests and observe expected failures.
- [x] Implement `answer_question` using `ask_model`, `load_index`, `OllamaEmbedder`, `search_index`.
- [x] Run tests until green.

### Task 3: CLI evaluation

**Files:** Update `deepseek-api/tests/test_rag_agent.py`; update `deepseek-api/rag_agent.py`.

- [x] Write tests for answer routing, atomic output, incomplete runs, resume, and explicit model choice.
- [x] Run tests and observe expected failures.
- [x] Implement `ask` and `evaluate` and JSON/Markdown outputs.
- [x] Run tests until green.

### Task 4: Live comparison and docs

**Files:** Create `docs/day-22/results.json`, `docs/day-22/comparison.md`; modify `deepseek-api/README.md`.

- [x] Document setup, commands, methodology, and limits.
- [x] Run all unit tests.
- [x] Run live evaluation with 20 successful model calls and 10 Ollama searches; the first seven pairs used GLM, the last three used DeepSeek at the user's suggestion.
- [x] Read the relevant book chapters and grade each answer manually.
- [x] Verify report totals and output contents.
