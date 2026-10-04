# Day 23 Relevance Filter and Query Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Top-20 → relevance filter → Top-5 retrieval, optional query rewrite, and a reproducible three-mode comparison for the existing book RAG.

**Architecture:** Keep the day-21 SQLite index and day-22 CLI intact. Put pure filtering and rewrite in `rag_relevance.py`, share the day-22 answer prompt from `rag_agent.py`, and put day-23 orchestration, CLI, and reporting in `rag_day23.py`.

**Tech Stack:** Python 3 standard library and `unittest`, existing Ollama `/api/embed` index, existing `providers.ask_model` for rewrite and answer.

**Design reference:** `docs/superpowers/specs/2026-10-04-day-23-relevance-filter-design.md`.

---

## File map

| File | Responsibility |
|---|---|
| `deepseek-api/rag_relevance.py` | Validate settings, filter scored candidates, rewrite one query |
| `deepseek-api/rag_agent.py` | Expose common answer generation while preserving day-22 behavior |
| `deepseek-api/rag_day23.py` | Run three retrieval modes, CLI, checkpointed evaluation, report |
| `deepseek-api/tests/test_rag_relevance.py` | Filter and rewrite unit tests |
| `deepseek-api/tests/test_rag_day23.py` | Orchestration, errors, evaluation and CLI tests |
| `docs/day-23/results.json`, `docs/day-23/comparison.md` | Real run and human assessment, created after code verification |
| `deepseek-api/README.md` | Day-23 usage and method |

## Task 1: Pure filter and rewrite

**Files:** Create `deepseek-api/tests/test_rag_relevance.py`; create `deepseek-api/rag_relevance.py`.

- [ ] Add table-driven tests with scores `[0.62, 0.58, 0.54, 0.39]`: `min_similarity=0.40`, `max_score_gap=0.08`, `top_k_after=5` keeps the first three; `top_k_after=2` keeps two. Equality at 0.40 and at `best-0.08` must pass. Assert input list is unchanged and output keeps its order.
- [ ] Add tests for empty candidates, no survivors, invalid Top-K and non-finite/bool scores or settings. Run `python3 -m unittest discover -s deepseek-api/tests -p test_rag_relevance.py -v` and observe missing-module failures.
- [ ] Implement `select_chunks` with the exact predicate `score >= min_similarity and score >= best_score - max_score_gap`, then truncate to `top_k_after`. Use `max()` for `best_score`; do not assume the caller has sorted candidates when applying the predicate, but preserve caller order.
- [ ] Test `rewrite_query` with a stub `ask` that records the payload (`temperature=0`, selected model, short system instruction, original question) **and the call keyword argument** `max_tokens=120`. `providers.ask_model` ignores an extra `max_tokens` key inside the payload, so assert `ask.call_args.kwargs["max_tokens"] == 120`. Test trimmed one-line output and failure on empty/multiline/non-string response; implement it without calling Ollama.
- [ ] Run the focused test file until green; commit this self-contained change.

## Task 2: Share the answer prompt without changing day 22

**Files:** Modify `deepseek-api/rag_agent.py`; modify `deepseek-api/tests/test_rag_agent.py`.

- [ ] Add a test that the extracted `generate_answer(question, chunks, *, model, ask)` produces the same payload as the current RAG path: original question, numbered sources, section, `chunk_id`, at most 1800 characters per excerpt, unchanged system instruction and temperature. Test the plain path with an empty `chunks` list.
- [ ] Run `python3 -m unittest discover -s deepseek-api/tests -p test_rag_agent.py -v` and confirm the new test fails before implementation.
- [ ] Extract the current prompt construction and provider response validation into `generate_answer`. Have existing `answer_question` call it after retrieval; keep its public signature, result keys and original CLI behavior.
- [ ] Run the focused tests, then `python3 -m unittest discover -s deepseek-api/tests -p test_document_index.py -v`; commit only the shared helper and tests.

## Task 3: Three retrieval modes

**Files:** Create `deepseek-api/tests/test_rag_day23.py`; create `deepseek-api/rag_day23.py`.

- [ ] Add tests with fake `search_index`, embedder, rewriter and answer model. All modes ask the index for `top_k=20`; `baseline` passes the first five; `filtered` applies the predicate; `rewritten` calls rewrite exactly once and searches with the rewritten text. In every mode, answer generation receives the original question.
- [ ] Test zero survivors both when search itself is empty (`baseline`) and when the filter removes all candidates (`filtered`): fixed insufficient-evidence answer, `answer_status="no_evidence"`, no answer-model call, zero citations. Test invalid mode and Ollama/index/provider errors become user-facing `RagError` messages. Confirm failures before implementation.
- [ ] Test public API and CLI boundaries for `top_k_before >= top_k_after > 0`, `-1 <= min_similarity <= 1`, `0 <= max_score_gap <= 2`, rejecting bool, NaN and infinity. Run with `--top-k-before 3 --top-k-after 2` to prove settings reach retrieval and truncation.
- [ ] Implement `answer_question_day23` returning `question`, `mode`, `search_query`, `candidates_before`, `chunks_after`, `answer`, `answer_status`, `model`, settings and index metadata. Strip `vector` and unused SQLite fields before returning/saving a candidate; preserve source, section, `chunk_id`, score and text.
- [ ] Implement CLI `ask` and validate `--top-k-before`, `--top-k-after`, `--min-similarity`, `--max-score-gap`; keep `--index` and `--ollama-url` as global options. Make the CLI print original question, `search_query`, answer, before/after counts, scores and source IDs; test both original and rewritten query display.
- [ ] Run the focused tests and three `--help` smoke checks (`rag_day23.py --help`, `ask --help`, `evaluate --help`); commit.

## Task 4: Repeatable comparison and report

**Files:** Extend `deepseek-api/tests/test_rag_day23.py`; extend `deepseek-api/rag_day23.py`.

- [ ] Add tests for ten-control-file validation, deterministic order `baseline → filtered → rewritten` per question, expected chapter detection in `chunks_after`, identical answer model across modes, and saved rewrite text. Tests should use temporary files and fake provider/search; no live API calls.
- [ ] Add tests for atomic final outputs, refusal to overwrite existing outputs without `--overwrite`, `.partial` after one completed question, safe resume, and rejection when index/questions/settings/model/**Ollama URL** changed. Inject a failure after `results.json` is written but before `comparison.md`: confirm the checkpoint remains and rerun recreates only the missing report. Inject a failure after both final files exist but before `.partial` is deleted: confirm rerun validates the completed data, removes only `.partial`, and leaves a manually modified report untouched. Run the focused tests and observe failures.
- [ ] Implement `evaluate` using `docs/day-22/control_questions.json`; hash question JSON and index bytes. Save `.partial` only after all three modes for a question finish. Include `schema_version`, `prompt_version`, answer model, embedding model, Ollama URL, Top-K and thresholds in settings. On resume, require exact settings and input fingerprints; unlike day 22, do not permit a different answer model within the same run. For finalization, write complete JSON first, report second, and delete `.partial` last. On recovery, verify the existing JSON's controls/settings/fingerprints against `.partial` before writing a missing report or, if both outputs already exist, before deleting the stale `.partial` without changing the report.
- [ ] Generate `docs/day-23/comparison.md` as a **draft** with per-question, per-mode original/search query, ordered source IDs and scores before/after, counts, expected-source hit, blank completeness cells (0–2), blank groundedness cells (`да`/`частично`/`нет`) and an explicit limitations section. Assert these fields in tests. Do not claim an improvement before manual grading.
- [ ] Run `python3 -m unittest discover -s deepseek-api/tests -p test_rag_day23.py -v`, then the full suite `python3 -m unittest discover -s deepseek-api/tests -v`; commit.

## Task 5: Live run, demo case and documentation

**Files:** Modify `deepseek-api/README.md`; create `docs/day-23/results.json` and `docs/day-23/comparison.md` through the CLI; update `docs/day-23/demo-testcase.md` only if actual commands or observations differ from the planned interface.

- [ ] Add README setup, CLI examples, the filter formula, Top-K defaults, key requirements, output paths, overwrite/resume behavior, and note that cosine score is not a probability.
- [ ] With Ollama and the API key available, run `python3 deepseek-api/rag_day23.py evaluate --model glm-4.7-flash` (or choose `deepseek-v4-flash` for the entire new run). Keep one answer model throughout; on a provider outage, resume with the same model.
- [ ] Inspect `results.json`: ten questions × three modes, every `baseline` has at most 5 chunks, every filtered mode has at most 5, candidates have at most 20, rewrite is recorded, vector values are absent, and stored settings match the command.
- [ ] Manually grade completeness 0–2 against each `expected_theses` list and groundedness as `да` (all claims supported), `частично` (some claims unsupported), or `нет` (essential claims unsupported) against saved chunk text. Fill the report with reviewer/date/status `проверено`, totals, concrete improvements/regressions, and whether expected chapters survived filtering. Compare `baseline` with `filtered`, then `filtered` with `rewritten`.
- [ ] Run the two separate out-of-book probes from the spec. Record if each actually abstains; count a substantive answer as a filter failure, without retuning the fixed threshold after seeing the evaluation. Do not include these probes in the ten-question completeness total.
- [ ] Perform the demo from `docs/day-23/demo-testcase.md`; record observed scores/counts in its results section or explain any variation. Run the full suite again and check `git diff --check` before claiming completion; commit docs and real output.

## Completion criteria

- `ask` exposes and honors all three modes and both Top-K values.
- The second stage enforces the declared threshold and may return fewer than five chunks.
- Rewrite affects retrieval only; the answer still addresses the original question.
- A same-model, same-index, same-question comparison is saved separately from day 22.
- The report includes human-checked answer quality and honestly records both gains and losses.
