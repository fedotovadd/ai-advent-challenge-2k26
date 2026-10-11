# Day 28 Local RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing citation checked RAG with local Qwen and record a resumable three pass comparison against DeepSeek when its key is available.

**Architecture:** Keep Day 24 retrieval, reranking, prompt, and validation as the single source of truth. Extract context preparation and generation into callable steps. Day 28 stores one immutable context snapshot per question and appends each measured attempt atomically.

**Tech Stack:** Python 3.14, unittest, SQLite index, Ollama, sentence-transformers, OpenAI client.

---

### Task 1: Local safety and Day 24 reusable steps

**Files:** `deepseek-api/deepseek_app/rag/rag_citations.py`, `deepseek-api/deepseek_app/rag/rag_reranker.py`, `deepseek-api/deepseek_app/rag/rag_day24.py`, `deepseek-api/tests/test_rag_day24.py`, `deepseek-api/tests/test_rag_reranker.py`

- [ ] Add failing tests: local model accepted, cloud suffix rejected, loopback URL required, reranker loads from local files only, provider gets JSON mode, context built once and reusable, invalid JSON raises after one repair.
- [ ] Run focused tests and confirm expected failures.
- [ ] Add model allowlist; loopback validator; opt-in local-only reranker; `prepare_context` and `generate_from_context` with timing and one repair; keep `answer_question_day24` API and old cloud behavior.
- [ ] Run focused and existing Day 24 tests.

### Task 2: Resumable pair runner

**Files:** `deepseek-api/deepseek_app/rag/rag_day28.py`, `deepseek-api/tests/test_rag_day28.py`

- [ ] Add failing tests for one retrieval per question, identical context, three passes, no cloud calls without key, per-attempt checkpoint, recovery with matching index/settings, and error preservation.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement `compare` CLI and runner. Hash index and questions; checkpoint snapshot before generation and each attempt after completion. Preserve raw model output and validation details. Resume only matching settings and index.
- [ ] Build report from observed data with per-stage times, medians/maxima, pass counts, explicit manual assessment fields and limitations.
- [ ] Run focused tests.

### Task 3: Documentation and live verification

**Files:** `deepseek-api/README.md`, `docs/day-28/results.json`, `docs/day-28/comparison.md`

- [ ] Document model/dependency preparation, local ask, comparison, recovery and manual review.
- [ ] Run all relevant unit tests.
- [ ] Check local Ollama and reranker files. Run one local ask without cloud keys; run three pass comparison if prerequisites are present. If prerequisites are missing, preserve the concrete error and do not invent measurements.
- [ ] Inspect report, saved JSON, and repository diff against every acceptance criterion.
