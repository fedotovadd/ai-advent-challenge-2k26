# Day 23 Cross-Encoder Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Query → bi-encoder Top-20 → real cross-encoder rerank/threshold → Top-5 → LLM, with rewrite between retrieval and reranking and a three-mode comparison.

**Architecture:** Reuse the day-21 index and day-22 answer prompt. A separate module loads the multilingual cross-encoder lazily, scores `(query, chunk)` pairs, and filters/sorts them. Day-23 orchestration keeps one first-stage candidate pool per mode, optionally rewrites the query after retrieval, and saves an auditable comparison.

**Tech Stack:** Python 3, `unittest`, Ollama, `sentence-transformers` with PyTorch, existing `providers.ask_model`.

**Design reference:** `docs/superpowers/specs/2026-10-04-day-23-relevance-filter-design.md`.

---

## Task 1: Cross-encoder adapter and pure rerank

**Files:** Create `deepseek-api/tests/test_rag_reranker.py`, `deepseek-api/rag_reranker.py`, `deepseek-api/requirements-rerank.txt`.

- [ ] Write a failing test for `rerank_candidates(query, candidates, scorer, *, top_k_after, min_rerank_score)` with fake normalized scores that reverse the cosine order. Assert pair input includes title, section and text, output order follows cross-encoder scores, and returned rows carry both scores and both ranks.
- [ ] Run `python3 -m unittest discover -s deepseek-api/tests -p test_rag_reranker.py -v` and confirm the failure is due to missing behavior.
- [ ] Implement the minimal pure function: finite normalized-score/count validation, stable tie-break by original rank and ID, threshold `>=`, Top-K truncation, no mutation of inputs. Cover empty input, all rejected, threshold equality, invalid settings and non-finite values with further red/green tests.
- [ ] Add `CrossEncoderReranker` with lazy `sentence_transformers.CrossEncoder(model_name, max_length=512)` and `predict(pairs, batch_size=..., activation_fn=torch.nn.Sigmoid())`; test using a fake model factory. Missing dependency/model load errors must name `requirements-rerank.txt`, never silently use cosine.
- [ ] Add `-r requirements.txt` plus `sentence-transformers>=6,<7` to optional requirements. Run focused tests; commit this task.

## Task 2: Rewrite based on first-stage candidates

**Files:** Extend `deepseek-api/tests/test_rag_reranker.py`, `deepseek-api/rag_reranker.py`.

- [ ] Write a failing test for `rewrite_query(original_question, candidates, *, model, ask)`: fake `ask` must receive the original question and bounded, numbered candidate summaries; verify `ask(payload, max_tokens=120)` and temperature 0.
- [ ] Implement rewrite with a capped total summary length, untrusted-document instruction, trimmed single-line response, and errors for empty/multiline/too-long output. Cover provider failure without silent fallback.
- [ ] Run focused tests; commit.

## Task 3: Shared answer prompt and three modes

**Files:** Modify `deepseek-api/rag_agent.py`, `deepseek-api/tests/test_rag_agent.py`; create `deepseek-api/rag_day23.py`, `deepseek-api/tests/test_rag_day23.py`.

- [ ] Write a failing day-22 regression test for extracted `generate_answer(question, chunks, *, model, ask)` preserving the exact system/user payload and 1800-character excerpt limit. Extract it; run `test_rag_agent.py` to green.
- [ ] Write failing tests for `answer_question_day23`: every mode retrieves Top-20 using the **original** query; `baseline` takes cosine Top-5; `reranked` uses cross-encoder with original query; `rewritten` calls rewrite after retrieval and uses only its result for cross-encoder. All answer prompts use the original question. Inject fake search, scorer, rewriter and provider.
- [ ] Implement orchestration and CLI `ask` with defaults 20/5/0.50. Validate `top_k_before >= top_k_after > 0`, threshold in `[0,1]`, finite values, supported mode/model. Empty post-stage context returns fixed no-evidence answer without provider call. Preserve both score/rank fields and strip vectors from output.
- [ ] Test missing index/Ollama/rewrite/reranker/provider errors, empty baseline search, empty rerank result and CLI display of original/search query plus Top-20/Top-5 scores. Run focused tests and CLI `--help`; commit.

## Task 4: Checkpointed three-mode evaluation

**Files:** Extend `deepseek-api/rag_day23.py`, `deepseek-api/tests/test_rag_day23.py`.

- [ ] Write failing tests for ten-question order (`baseline`, `reranked`, `rewritten`), same model/first-stage settings, expected chapter detection, saved rewrite/rerank scores, draft report listing ordered source IDs and both scores, and manual grading placeholders.
- [ ] Write failing tests for existing-output protection, completed-question `.partial` checkpoint, settings/index/questions fingerprint mismatch, failure between JSON and report writes, and failure after both outputs before checkpoint deletion. Ensure recovery preserves a manually edited report.
- [ ] Implement `evaluate` with the day-22 control file, atomic writes and strict resume checks. The final JSON is written before draft Markdown, checkpoint removed last. Report the model IDs, threshold, Top-K, answer scores and limitations. Never claim an improvement before manual review.
- [ ] Run focused and full `unittest discover`; commit.

## Task 5: Live evaluation and demo

**Files:** Modify `deepseek-api/README.md`, `docs/day-23/demo-testcase.md`; create `docs/day-23/results.json`, `docs/day-23/comparison.md` through CLI.

- [ ] Document optional dependency installation, first model download, Ollama setup, CLI modes, score meaning and query rewrite position. Install optional requirements in a project virtual environment and check the model loads locally.
- [ ] Run `evaluate` with one answer model for all ten questions. If external services are unavailable, preserve passing offline tests and report the concrete blocker; do not fabricate scores or manual judgments.
- [ ] Check 10 × 3 results, Top-20/Top-5 limits, cross-encoder reorder evidence, no embedded vectors, source hits and recorded settings. Manually score completeness 0–2 and groundedness (`да`/`частично`/`нет`); mark the report `проверено` with date only after checking saved answers against source text.
- [ ] Run the two out-of-book probes separately and record whether the system abstains. Execute the demo case and fill its observed-result table. Run full tests and `git diff --check` before a completion claim; commit the work.

## Completion criteria

- The second stage is a genuine locally run cross-encoder on all retrieved query–chunk pairs, not a cosine threshold masquerading as reranking.
- `rewritten` derives its search formulation after Top-20 and uses it to rerank that fixed pool.
- Top-K and the cross-encoder cutoff are configurable and recorded.
- A reviewed report compares baseline, reranked, and rewritten answers honestly, including regressions.
