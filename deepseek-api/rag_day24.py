"""Citation-grounded Day 24 RAG CLI and evaluation."""

import math
from pathlib import Path

from document_index import DEFAULT_DB, OllamaEmbedder, load_index, search_index
from providers import ask_model
from rag_agent import MODEL, RagError
from rag_citations import (MODELS, NO_EVIDENCE_ANSWER, build_answer_payload,
                           validate_answer_json)
from rag_reranker import (DEFAULT_RERANKER_MODEL, CrossEncoderReranker,
                          rerank_candidates)


TOP_K_BEFORE = 20
TOP_K_AFTER = 5
MIN_RERANK_SCORE = 0.50
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = PROJECT_ROOT / "docs/day-22/control_questions.json"
DEFAULT_RESULTS = PROJECT_ROOT / "docs/day-24/results.json"
DEFAULT_REPORT = PROJECT_ROOT / "docs/day-24/evaluation.md"


def _validate(question, model, top_k_before, top_k_after, min_rerank_score):
    if not isinstance(question, str) or not question.strip():
        raise RagError("Укажите непустой вопрос")
    if model not in MODELS:
        raise RagError("Неизвестная модель")
    if (isinstance(top_k_before, bool) or not isinstance(top_k_before, int)
            or isinstance(top_k_after, bool) or not isinstance(top_k_after, int)
            or not top_k_before >= top_k_after > 0):
        raise RagError("Требуется top_k_before >= top_k_after > 0")
    if (isinstance(min_rerank_score, bool)
            or not isinstance(min_rerank_score, (int, float))
            or not math.isfinite(min_rerank_score)
            or not 0 <= min_rerank_score <= 1):
        raise RagError("Порог min_rerank_score должен быть конечным числом в [0, 1]")


def answer_question_day24(question, *, index_path=DEFAULT_DB,
                          ollama_url="http://localhost:11434", model=MODEL,
                          top_k_before=TOP_K_BEFORE, top_k_after=TOP_K_AFTER,
                          min_rerank_score=MIN_RERANK_SCORE,
                          reranker_model=DEFAULT_RERANKER_MODEL,
                          index_loader=load_index, embedder_factory=OllamaEmbedder,
                          search=search_index, scorer=None, ask=ask_model):
    """Search once, rerank, and release an answer only after quote validation."""
    _validate(question, model, top_k_before, top_k_after, min_rerank_score)
    question = question.strip()
    try:
        metadata, _ = index_loader(index_path)
        embedder = embedder_factory(metadata["model"], ollama_url)
        found = search(index_path, question, "structural", embedder,
                       top_k=top_k_before)
    except Exception as exc:
        raise RagError(f"Поиск: {exc}") from exc
    candidates = []
    for rank, row in enumerate(found, 1):
        candidate = {key: value for key, value in row.items() if key != "vector"}
        candidate["bi_score"] = row["score"]
        candidate["bi_rank"] = rank
        candidates.append(candidate)
    selected = []
    if candidates:
        try:
            selected = rerank_candidates(
                question, candidates,
                scorer if scorer is not None else CrossEncoderReranker(reranker_model),
                top_k_after=top_k_after, min_rerank_score=min_rerank_score)
        except Exception as exc:
            raise RagError(f"Reranker: {exc}") from exc
    result = {
        "question": question, "model": model, "temperature": 0,
        "index": {"path": str(index_path), "embedding_model": metadata["model"],
                  "epub_sha256": metadata["epub_sha256"], "strategy": "structural"},
        "settings": {"top_k_before": top_k_before, "top_k_after": top_k_after,
                     "min_rerank_score": min_rerank_score,
                     "reranker_model": reranker_model},
        "selected": selected,
    }
    if not selected:
        return {**result, "status": "no_evidence", "reason": "below_threshold",
                "answer": NO_EVIDENCE_ANSWER, "sources": [], "citations": []}
    payload = build_answer_payload(question, selected, model)
    for attempt in range(2):
        try:
            response = ask(payload)
        except Exception as exc:
            raise RagError(f"Провайдер ответа: {exc}") from exc
        content = response.get("content") if isinstance(response, dict) else None
        try:
            verified = validate_answer_json(content, selected)
        except ValueError as exc:
            if attempt:
                raise RagError(f"Ответ модели не прошёл проверку: {exc}") from exc
            payload = {**payload, "messages": [*payload["messages"], {
                "role": "user", "content": (
                    f"Исправь формат JSON и цитаты: {exc}. "
                    "Верни один JSON-объект по тем же фрагментам без дополнительного текста."
                )}]}
        else:
            if verified["status"] == "no_evidence":
                verified["reason"] = "model_abstained"
            return {**result, **verified}
