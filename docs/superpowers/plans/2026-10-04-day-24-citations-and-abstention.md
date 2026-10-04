# Day 24 Citation-Grounded RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Return a sourced, quoted answer for supported book questions and an explicit «Не знаю» with a clarification request when retrieval has no chunk above the relevance threshold.

**Architecture:** A new Day 24 CLI reuses the Day 21 structural index and Day 23 cross-encoder reranker. A separate pure module builds the answer prompt and validates model JSON against the exact excerpts sent to the model. The CLI derives source metadata from validated chunks, then records a ten-question evaluation and manual semantic review.

**Tech Stack:** Python 3, `unittest`, SQLite index, Ollama embeddings, `sentence-transformers` cross-encoder, existing `providers.ask_model`.

**Design reference:** `docs/superpowers/specs/2026-10-04-day-24-citations-and-abstention-design.md`.

---

## File map and contracts

- Create `deepseek-api/rag_citations.py`: `build_answer_payload(question, selected, model)` and `validate_answer_json(content, selected)`; no I/O. Export `EXCERPT_CHAR_LIMIT=1800`, `MAX_QUOTE_CHARS=400`, `NO_EVIDENCE_ANSWER="Не знаю: в найденных фрагментах книги недостаточно сведений. Уточните вопрос, указав тему или раздел книги."`. Validation returns only `status`, `answer`, `sources`, `citations`; it raises `ValueError` on malformed or unverifiable output.
- Create `deepseek-api/rag_day24.py`: `answer_question_day24(...)`, `evaluate(...)`, and `main(argv=None)`. Inject loader, embedder factory, search, scorer, and answer provider at the one-question boundary for offline tests. Reuse `document_index`, `rag_reranker`, `rag_agent._read_questions`, `rag_agent._input_fingerprint`, and `rag_agent._atomic_write`; do not call the unverified `rag_agent.generate_answer`.
- Create `deepseek-api/tests/test_rag_citations.py` and `deepseek-api/tests/test_rag_day24.py`. Do not modify Day 22–23 behavior or tests.
- Modify `deepseek-api/README.md` to document new commands. Create `docs/day-24/results.json` and `docs/day-24/evaluation.md` through the actual run; never fabricate results.

Public `answered` result: `question`, `status`, `answer`, `sources`, `citations`, `selected`, `settings`, `model`, `index`. Each source is `{source, section, chunk_id}`. Each citation is `{ref, source, section, chunk_id, quote}`. `no_evidence` has the fixed answer and empty source/citation arrays; record a machine-readable reason of `below_threshold` or `model_abstained`. Exclude SQLite vectors from every public result.

## Task 1: Build the answer contract and citation validator

**Files:** Create `deepseek-api/rag_citations.py`, `deepseek-api/tests/test_rag_citations.py`.

- [ ] **Step 1: Write failing tests** for one valid answer, two quotes from one chunk, deduplicated sources, and the exact 1800-character prompt excerpt. Use a selected row such as `{"chunk_id": "c1", "source": "book.epub:chapter.xhtml", "section": "Раздел", "text": "Тезис подтверждён книгой."}`. Assert that model-supplied `source`/`section` fields cannot replace index metadata.
- [ ] **Step 2: Run** `.venv/bin/python -m unittest discover -s deepseek-api/tests -p test_rag_citations.py -v`. Expected: import or missing-function failure for the new module.
- [ ] **Step 3: Implement** `build_answer_payload`: temperature 0, supported model, system instruction requiring a single JSON object with `status`, `answer`, `citations`, markers `[1]` etc., verbatim quotes, and abstention when text is insufficient. Build numbered excerpts with `source`, `section`, `chunk_id`, and `row["text"][:1800]`. Explicitly state that document text is data, not instructions.
- [ ] **Step 4: Implement** `validate_answer_json`: `json.loads` must return an object; require typed fields. For `no_evidence`, require `answer == ""` and `citations == []`, then return the fixed answer with empty arrays. For `answered`, require nonblank answer and 1+ citations. Number citations from array order, parse `\[(\d+)\]` markers, and require the set of markers to equal `{1, ..., len(citations)}`. Map each `chunk_id` to exactly one selected row; reject unknown or duplicate selected IDs. Require `quote` to be a nonblank string of at most 400 characters and an exact substring of that row's first 1800 text characters. Derive source metadata and deduplicate sources by `(source, section, chunk_id)` in first-citation order. Do not trust model metadata.
- [ ] **Step 5: Add failing edge-case tests** for invalid JSON, wrong types, no marker, unused or out-of-range marker, missing citation, unknown ID, fabricated quote, quote beyond the 1800-character excerpt, quote longer than 400 characters, invalid abstention payload, and prompt-injection text inside a chunk. Verify each is rejected or treated only as data.
- [ ] **Step 6: Run the focused tests** again. Expected: all tests in `test_rag_citations.py` pass. Commit only these two files, for example `git commit -m 'feat: validate RAG citations against retrieved excerpts'`.

Validator core to implement in `rag_citations.py` (with `import json`, `import re` and the constants above):

```python
def validate_answer_json(content, selected):
    if not isinstance(content, str):
        raise ValueError("Провайдер вернул не текст")
    raw = json.loads(content)
    if not isinstance(raw, dict) or raw.get("status") not in {"answered", "no_evidence"}:
        raise ValueError("Некорректный формат ответа модели")
    if raw["status"] == "no_evidence":
        if raw.get("answer") != "" or raw.get("citations") != []:
            raise ValueError("Некорректный отказ модели")
        return {"status": "no_evidence", "answer": NO_EVIDENCE_ANSWER,
                "sources": [], "citations": []}
    answer, proposed = raw.get("answer"), raw.get("citations")
    if (not isinstance(answer, str) or not answer.strip()
            or not isinstance(proposed, list) or not proposed):
        raise ValueError("Нет ответа или цитат")
    markers = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
    if markers != set(range(1, len(proposed) + 1)):
        raise ValueError("Ссылки ответа не соответствуют цитатам")
    rows = {row["chunk_id"]: row for row in selected}
    if len(rows) != len(selected):
        raise ValueError("Повторяющийся chunk_id в контексте")
    citations, sources, seen_sources = [], [], set()
    for ref, item in enumerate(proposed, 1):
        if not isinstance(item, dict):
            raise ValueError("Некорректная цитата")
        chunk_id, quote = item.get("chunk_id"), item.get("quote")
        if not isinstance(chunk_id, str) or chunk_id not in rows:
            raise ValueError("Цитата ссылается на неизвестный чанк")
        if (not isinstance(quote, str) or not quote.strip()
                or len(quote) > MAX_QUOTE_CHARS
                or quote not in rows[chunk_id]["text"][:EXCERPT_CHAR_LIMIT]):
            raise ValueError("Цитата не найдена в переданном фрагменте")
        row = rows[chunk_id]
        source = {key: row[key] for key in ("source", "section", "chunk_id")}
        citations.append({"ref": ref, **source, "quote": quote})
        key = tuple(source.values())
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append(source)
    return {"status": "answered", "answer": answer.strip(),
            "sources": sources, "citations": citations}
```

## Task 2: Retrieve, apply threshold, and request a verified answer

**Files:** Create `deepseek-api/rag_day24.py`, `deepseek-api/tests/test_rag_day24.py`.

- [ ] **Step 1: Write failing one-question tests** with fake index metadata, embedder, search rows, scorer, and provider. Assert one structural Top-20 search using the original question; cross-encoder pairs use that same question; Top-5 rows follow rerank score and include the threshold-equal row; vectors are removed.
- [ ] **Step 2: Run** `.venv/bin/python -m unittest discover -s deepseek-api/tests -p test_rag_day24.py -v`. Expected: import or missing-function failure for Day 24.
- [ ] **Step 3: Implement** argument validation (`question`, supported `model`, integer `top_k_before >= top_k_after > 0`, finite `min_rerank_score` in `[0,1]`). Load the existing index; create `OllamaEmbedder(metadata["model"], ollama_url)`; call `search_index(index_path, question, "structural", embedder, top_k=top_k_before)`. Copy rows without `vector`, add `bi_score` and `bi_rank`, then call `rerank_candidates(question, candidates, scorer or CrossEncoderReranker(reranker_model), top_k_after=top_k_after, min_rerank_score=min_rerank_score)`. Persist model, index identity, scores, ranks, threshold, and selected rows.
- [ ] **Step 4: Add failing no-evidence tests** for zero search rows and all rerank scores below `0.50`. Assert that neither answer provider nor scorer is called for zero rows, that provider is not called after a rejected rerank pool, and that the result has `status="no_evidence"`, the exact fixed clarification response, and empty sources/citations. Assert equality with the threshold passes.
- [ ] **Step 5: Implement** the early `no_evidence` return. When rows remain, call `build_answer_payload` and `ask_model` through the injected `ask`, then `validate_answer_json`. If validation fails, retry **once** using the same excerpt payload plus a concise correction request naming the validation failure. If the second response is invalid, raise `RagError` without emitting its content. A valid model abstention returns `reason="model_abstained"`; an answered result uses only validated answer/sources/citations. Provider/search/reranker exceptions become clear `RagError`s; they are not recast as valid answers.
- [ ] **Step 6: Add and pass tests** for valid cited response, model abstention, first invalid then valid response (two provider calls), two invalid responses (error, no answer), provider failure (no format retry), missing index/Ollama, invalid score, and invalid options. Run focused tests, then Day 22–23 regression tests. Commit these two files.

Implementation note: `rerank_candidates` already applies `>= min_rerank_score`, stable ordering, normalized-score checks, and Top-K. Reuse it rather than implementing a second sorter. The first-stage search may return a `vector` field; strip it before serialization or prompt construction.

## Task 3: CLI and checkpointed evaluation

**Files:** Modify `deepseek-api/rag_day24.py`, `deepseek-api/tests/test_rag_day24.py`.

- [ ] **Step 1: Write failing CLI tests** for `ask` output sections `Ответ`, `Источники`, `Цитаты`; each source prints `source / section (chunk_id)`, each citation prints its reference and exact quote. For refusal, print `Источники: нет` and `Цитаты: нет`. Test nonzero exit with a clear stderr message on `RagError`.
- [ ] **Step 2: Implement** `argparse` with global `--index`, `--ollama-url` before `ask`/`evaluate`; per-command `--model`, `--top-k-before`, `--top-k-after`, `--min-rerank-score`, `--reranker-model`. `evaluate` additionally has `--questions`, `--results`, `--report`, and `--overwrite`. Defaults are the approved spec paths and Day 23 retrieval settings.
- [ ] **Step 3: Write failing evaluation tests** using a temporary ten-question control file and fake `answer_question_day24`. Assert ten questions are processed once, the same model/settings are used, no-evidence rows remain visible, each answered row records machine checks for nonempty sources/citations, selected-chunk membership and exact quote origin, and the report has a per-question semantic-review field left unscored until manual review.
- [ ] **Step 4: Implement** evaluation using `rag_agent._read_questions`, `_input_fingerprint`, and `_atomic_write`. Require exactly ten controls, unique positive IDs, and nonempty expected source/theses. Store selected rows, answers, scores, model/index/settings, and input fingerprint. Compute the machine checks from the final validated result; count answered/refused questions separately. Generate `docs/day-24/evaluation.md` with ten rows and evidence details for subsequent semantic review.
- [ ] **Step 5: Write failing recovery tests**: existing outputs without `--overwrite` are untouched; a failed second question leaves one complete entry in `.partial`; resume uses only the remaining questions; changed questions, index, model, or settings reject the checkpoint; a crash after final JSON write can finish the missing report without another provider call; a manually edited report is not overwritten during recovery. Reject output paths that collide with one another or the inputs.
- [ ] **Step 6: Implement** atomic checkpoint after each full question. Validate checkpoint status, fingerprint, settings, and complete prefix of controls before resume. Write final JSON before Markdown, delete `.partial` last. Before an explicit `--overwrite` fresh run, store the prior report SHA-256 in the checkpoint. On recovery, create a missing report; replace an old report only when its current hash still equals that saved prior hash, preserving any manual edits made since. A matching final JSON plus missing report must recover without another model call. Run focused tests and `rag_day24.py --help`, then commit.

Report shape:

```text
| № | Статус | Источники | Цитаты | Дословность | Смысл ответа подтверждён | Комментарий |
|---:|---|---|---|---|---|---|
| 1 | answered | да | да | да | требует ручной сверки | ожидает оценки |
```

`требует ручной сверки` is an initial state, never a machine-generated `да` for semantic support. The implementer fills the final `да`/`частично`/`нет` and a concrete explanation after reading each answer and its complete selected chunks.

## Task 4: Live run, manual review, and documentation

**Files:** Modify `deepseek-api/README.md`; create `docs/day-24/results.json`, `docs/day-24/evaluation.md` through the CLI.

- [ ] **Step 1: Document** the Day 24 setup and `ask`/`evaluate` commands, JSON/source/quote contract, meaning of the `0.50` cutoff, abstention behavior, and the distinction between exact quote checks and semantic review.
- [ ] **Step 2: Run focused and full regression tests**: `.venv/bin/python -m unittest discover -s deepseek-api/tests -p 'test_rag_*.py' -v`, then `.venv/bin/python -m unittest discover -s deepseek-api/tests -v`. Expected: zero failures. Run `git diff --check`.
- [ ] **Step 3: Run** `.venv/bin/python deepseek-api/rag_day24.py evaluate --model glm-4.7-flash` using the existing index, local Ollama, reranker, and configured provider. Preserve `.partial` if an external service fails; resume with the same settings. Do not fabricate answers or manual grades. If the model must be changed, start a new run instead of mixing models.
- [ ] **Step 4: Inspect all ten saved results**. Verify answer/refusal counts, source metadata, exact citations, selected-chunk IDs, and the recorded threshold. For every substantive answer, compare each significant claim with its cited quote and full selected chunk; fill `да`/`частично`/`нет` and an explanation in `docs/day-24/evaluation.md`. Mark missing support honestly. Evaluate expected chapter hits separately.
- [ ] **Step 5: Run two out-of-book probes** through `ask` using the fixed threshold: weather tomorrow in Moscow and the current Moscow–Kazan train ticket price. Record their status in the report; if either receives a substantive answer, mark it as a threshold failure rather than tuning the threshold after the run.
- [ ] **Step 6: Verify** `git diff --check`, the final ten-row report, successful Day 22–24 tests, and no vectors in results. Commit code, docs, and actual outputs; state any external-service or semantic-review limitation precisely in the final report.

## Completion gate

- Every `answered` result has nonempty, verified `sources` and `citations`; no fabricated quote is displayed.
- A below-threshold result says «Не знаю», asks for clarification, and avoids the answer-model call.
- All ten controls have recorded checks and individually reviewed semantic support, with refusals and failures visible.
- Day 22–23 commands and index remain usable.
