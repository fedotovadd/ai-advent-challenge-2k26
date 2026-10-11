"""Build, compare and search a local EPUB embedding index."""

import argparse
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from deepseek_app.rag.epub_corpus import EXCLUDED, read_epub
from deepseek_app.rag.index_chunking import make_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EPUB = PROJECT_ROOT / "docs/day-21/corpus/agentic-design-patterns-ru.epub"
DEFAULT_DB = PROJECT_ROOT / "docs/day-21/index.sqlite3"
DEFAULT_REPORT = PROJECT_ROOT / "docs/day-21/comparison.md"
SCHEMA_VERSION = 1
QUERIES = (
    "Как агенту хранить память проекта между сессиями?",
    "Как проверять изменения кода перед коммитом?",
    "Зачем разделять планирование и выполнение задачи?",
)


class IndexErrorMessage(Exception):
    """Expected user-facing indexing failure."""


class OllamaEmbedder:
    def __init__(self, model="embeddinggemma", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, texts):
        payload = json.dumps({"model": self.model, "input": texts, "truncate": False}).encode()
        request = Request(self.base_url + "/api/embed", payload,
                          {"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=120) as response:
                result = json.load(response)
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")[:300]
            if exc.code == 404:
                raise IndexErrorMessage(f"Модель {self.model!r} не найдена в Ollama. Выполните: ollama pull {self.model}. {details}") from exc
            raise IndexErrorMessage(f"Ollama API: HTTP {exc.code}: {details}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise IndexErrorMessage(f"Ollama недоступна по адресу {self.base_url}. Запустите `ollama serve`: {exc}") from exc
        except (ValueError, TypeError) as exc:
            raise IndexErrorMessage(f"Некорректный ответ Ollama: {exc}") from exc
        return result.get("embeddings") if isinstance(result, dict) else None


def _validate_vectors(vectors, count, dimension=None):
    if not isinstance(vectors, list) or len(vectors) != count:
        raise IndexErrorMessage(f"Ollama вернула {len(vectors) if isinstance(vectors, list) else 'неверное число'} векторов вместо {count}")
    for vector in vectors:
        if not isinstance(vector, list) or not vector or (dimension and len(vector) != dimension):
            raise IndexErrorMessage("Неверная размерность вектора Ollama")
        if dimension is None:
            dimension = len(vector)
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
            raise IndexErrorMessage("Вектор Ollama содержит неверные значения")
        if not any(x != 0 for x in vector):
            raise IndexErrorMessage("Ollama вернула нулевой вектор")
    return dimension


def build_index(epub_path, db_path, embedder, limit=250, overlap=40, batch_size=16, progress=None):
    try:
        docs, digest = read_epub(epub_path)
    except ValueError as exc:
        raise IndexErrorMessage(str(exc)) from exc
    if batch_size <= 0:
        raise IndexErrorMessage("Размер пакета должен быть положительным")
    chunks = [*make_chunks(docs, digest, "fixed", limit, overlap),
              *make_chunks(docs, digest, "structural", limit, overlap)]
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=".index-", suffix=".sqlite3", dir=db_path.parent)
    os.close(handle)
    try:
        with sqlite3.connect(temp_name) as db:
            db.executescript("""
                CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE documents(source TEXT PRIMARY KEY, title TEXT NOT NULL, word_count INTEGER NOT NULL);
                CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, strategy TEXT NOT NULL,
                    source TEXT NOT NULL REFERENCES documents(source), title TEXT NOT NULL,
                    section TEXT NOT NULL, ordinal INTEGER NOT NULL, text TEXT NOT NULL,
                    word_count INTEGER NOT NULL, overlap_words INTEGER NOT NULL,
                    crosses_section INTEGER NOT NULL, vector TEXT NOT NULL);
                CREATE INDEX chunks_strategy ON chunks(strategy);
            """)
            for doc in docs:
                db.execute("INSERT INTO documents VALUES(?,?,?)", (doc.source, doc.title,
                           sum(len(block.text.split()) for block in doc.blocks if block.kind == "body")))
            dimension = None
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                vectors = embedder.embed([chunk.embedding_text for chunk in batch])
                dimension = _validate_vectors(vectors, len(batch), dimension)
                db.executemany("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?,?)", [
                    (chunk.chunk_id, chunk.strategy, chunk.source, chunk.title,
                     chunk.section, chunk.ordinal, chunk.text, chunk.word_count,
                     chunk.overlap_words, int(chunk.crosses_section), json.dumps(vector))
                    for chunk, vector in zip(batch, vectors)
                ])
                if progress:
                    progress(min(start + batch_size, len(chunks)), len(chunks))
            meta = {"schema_version": SCHEMA_VERSION, "epub_sha256": digest,
                    "epub_name": Path(epub_path).name, "model": embedder.model,
                    "dimension": dimension, "limit_words": limit,
                    "overlap_words": overlap, "batch_size": batch_size,
                    "document_count": len(docs), "chunk_count": len(chunks),
                    "source_words": sum(len(block.text.split()) for doc in docs for block in doc.blocks if block.kind == "body")}
            db.executemany("INSERT INTO metadata VALUES(?,?)", [(key, json.dumps(value)) for key, value in meta.items()])
        os.replace(temp_name, db_path)
        return meta
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_index(db_path):
    db_path = Path(db_path)
    if not db_path.is_file():
        raise IndexErrorMessage(f"Готовый индекс не найден: {db_path}. Выполните build.")
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            meta = {row["key"]: json.loads(row["value"]) for row in db.execute("SELECT key,value FROM metadata")}
            if meta.get("schema_version") != SCHEMA_VERSION or not all(key in meta for key in ("model", "dimension", "epub_sha256")):
                raise IndexErrorMessage("Несовместимая схема индекса; выполните build.")
            rows = [dict(row) for row in db.execute("SELECT * FROM chunks ORDER BY strategy,source,ordinal")]
            if not rows or len(rows) != meta.get("chunk_count"):
                raise IndexErrorMessage("Неполный индекс; выполните build.")
            return meta, rows
    except sqlite3.DatabaseError as exc:
        raise IndexErrorMessage(f"Повреждённая или несовместимая схема индекса: {exc}") from exc


def _cosine(a, b):
    numerator = sum(x * y for x, y in zip(a, b))
    denominator = math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))
    return numerator / denominator if denominator else 0.0


def search_index(db_path, query, strategy, embedder, top_k=5):
    if strategy not in {"fixed", "structural"}:
        raise IndexErrorMessage("Стратегия должна быть fixed или structural")
    if not query.strip() or top_k <= 0:
        raise IndexErrorMessage("Укажите непустой запрос и положительный top-k")
    meta, rows = load_index(db_path)
    if embedder.model != meta["model"]:
        raise IndexErrorMessage(f"Индекс создан моделью {meta['model']}; используйте её для поиска")
    vectors = embedder.embed([query])
    _validate_vectors(vectors, 1, meta["dimension"])
    query_vector = vectors[0]
    found = []
    for row in rows:
        if row["strategy"] == strategy:
            vector = json.loads(row["vector"])
            _validate_vectors([vector], 1, meta["dimension"])
            found.append({**row, "score": _cosine(query_vector, vector)})
    return sorted(found, key=lambda row: (-row["score"], row["chunk_id"]))[:top_k]


def compare_index(db_path, embedder, report_path=DEFAULT_REPORT):
    meta, rows = load_index(db_path)
    if embedder.model != meta["model"]:
        raise IndexErrorMessage(f"Индекс создан моделью {meta['model']}")
    lines = ["# День 21 — сравнение разбиения EPUB", "",
             f"Корпус: `{meta['epub_name']}`; SHA-256: `{meta['epub_sha256']}`.",
             f"Содержательных XHTML-файлов: {meta['document_count']}; слов основного текста: {meta['source_words']} (по пробелам).",
             f"Исключены служебные страницы: {', '.join(sorted(EXCLUDED))}.",
             f"Модель: `{meta['model']}`; размерность: {meta['dimension']}; лимит: {meta['limit_words']} слов; перекрытие: {meta['overlap_words']} слов.",
             "", "| Стратегия | Чанков | Средняя длина | Медиана | Максимум | Повтор слов | Пересечений разделов |", "|---|---:|---:|---:|---:|---:|---:|"]
    for strategy in ("fixed", "structural"):
        subset = [row for row in rows if row["strategy"] == strategy]
        lengths = [row["word_count"] for row in subset]
        lines.append(f"| {strategy} | {len(subset)} | {statistics.mean(lengths):.1f} | {statistics.median(lengths):g} | {max(lengths)} | {sum(row['overlap_words'] for row in subset)} | {sum(row['crosses_section'] for row in subset)} |")
    lines += ["", "## Одинаковые запросы", ""]
    for query in QUERIES:
        lines.append(f"### {query}")
        lines.append("")
        for strategy in ("fixed", "structural"):
            results = search_index(db_path, query, strategy, embedder, top_k=3)
            lines.append(f"**{strategy}:**")
            lines.append("")
            for row in results:
                snippet = row["text"][:150].replace("|", "\\|").replace("\n", " ")
                lines.append(f"- {row['score']:.4f} — {row['title']} / {row['section']} (`{row['source']}`, `{row['chunk_id']}`): {snippet}…")
            lines.append("")
    lines += ["## Наблюдение", "",
              "Фиксированное окно может захватывать соседние разделы, тогда как структурное сохраняет границы H1–H3. Короткие разделы дают более короткие структурные чанки. Эти запросы показывают конкретные результаты, но не доказывают общего превосходства стратегии.", ""]
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Локальный индекс EPUB на Ollama embeddinggemma")
    parser.add_argument("--index", type=Path, default=DEFAULT_DB)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Построить индекс")
    build.add_argument("--epub", type=Path, default=DEFAULT_EPUB)
    build.add_argument("--model", default="embeddinggemma")
    build.add_argument("--batch-size", type=int, default=16)
    compare = sub.add_parser("compare", help="Сохранить сравнение стратегий")
    compare.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    search = sub.add_parser("search", help="Поиск по индексу")
    search.add_argument("query")
    search.add_argument("--strategy", choices=("fixed", "structural"), required=True)
    search.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            embedder = OllamaEmbedder(args.model, args.ollama_url)
            meta = build_index(args.epub, args.index, embedder, batch_size=args.batch_size,
                               progress=lambda done, total: print(f"Эмбеддинги: {done}/{total}", flush=True))
            print(f"Готово: {meta['document_count']} глав, {meta['chunk_count']} чанков, {meta['source_words']} слов; {args.index}")
        else:
            meta, _ = load_index(args.index)
            embedder = OllamaEmbedder(meta["model"], args.ollama_url)
            if args.command == "compare":
                report = compare_index(args.index, embedder, args.report)
                print(report.split("## Одинаковые запросы")[0])
                print(f"Отчёт: {args.report}")
            else:
                for row in search_index(args.index, args.query, args.strategy, embedder, args.top_k):
                    print(f"{row['score']:.4f}  {row['title']} / {row['section']}\n  {row['source']}  {row['chunk_id']}\n  {row['text'][:180]}…")
        return 0
    except (IndexErrorMessage, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
