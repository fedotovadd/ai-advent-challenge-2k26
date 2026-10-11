# Day 29 Local LLM Optimization Implementation Plan

> **For agentic workers:** Implement the tasks in order, using failing tests before production changes.

**Goal:** Compare reproducible local Qwen RAG configurations, select a measured profile, and expose it through a CLI.

**Architecture:** Day 29 reuses Day 24 context preparation and Day 28 baseline prompt, but owns immutable snapshots, candidate profiles, checkpointed generation, measurements, scoring, and reporting. The provider adapter explicitly allowlists all local aliases. The selected profile is a JSON artifact consumed by `ask`.

**Tech Stack:** Python 3, OpenAI compatible Ollama API, standard library `unittest`, existing RAG modules.

---

### Task 1: Local provider routing and metadata

- [x] Write tests for allowlisted local aliases, unknown variant rejection, and prompt token counts.
- [x] Verify failures, then implement the smallest provider change.
- [x] Run provider tests.

### Task 2: Day 29 experiment core

- [x] Write tests for profile parameters, same saved snapshot, status classification, tokens, and strict resume checks.
- [x] Verify failures, then add `rag_day29.py` with profile loading, prompt rendering, snapshot creation, checkpointed calls, and CLI commands.
- [x] Run Day 29 tests.

### Task 3: Measurements and report

- [x] Write tests for summary math and output on incomplete measurements.
- [x] Implement Ollama identity/context checks, unload/warm protocol, RSS sampling, and paired report.
- [x] Run relevant tests.

### Task 4: Live experiment and selection

- [x] Verify installed model, digest, context allocation, quantization, machine resources, and availability of same-version higher-precision weights.
- [x] Run baseline and all applicable candidate blocks on saved snapshots; keep every response and measurement.
- [x] Score responses against Day 28 rubric before creating the final table; select a valid candidate.
- [x] Run a fresh alternating paired comparison, save final profile and report.
- [x] Smoke test `ask` with evidence, empty context, and unavailable Ollama behavior.

### Task 5: Documentation and final verification

- [x] Add reproduction commands and measurement caveats to README.
- [x] Run full relevant tests and inspect result counts, hashes, profile, and git diff.
