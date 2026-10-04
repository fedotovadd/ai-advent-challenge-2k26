"""Deterministic fixed and heading-aware whitespace word chunking."""

from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    strategy: str
    source: str
    title: str
    section: str
    ordinal: int
    text: str
    word_count: int
    crosses_section: bool
    overlap_words: int

    @property
    def embedding_text(self):
        return f"Глава: {self.title}\nРаздел: {self.section}\n\n{self.text}"


def _tokens(doc):
    headings = {}
    words = []
    for block in doc.blocks:
        if block.kind == "heading":
            headings[block.level] = block.text
            headings = {level: name for level, name in headings.items() if level <= block.level}
        else:
            section = " / ".join(headings[level] for level in sorted(headings)) or doc.title
            words.extend((word, section) for word in block.text.split())
    return words


def make_chunks(docs, epub_sha, strategy, limit=250, overlap=40):
    if strategy not in {"fixed", "structural"} or limit <= 0 or not 0 <= overlap < limit:
        raise ValueError("Неверная стратегия или параметры разбиения")
    chunks = []
    for doc in docs:
        tokens = _tokens(doc)
        if not tokens:
            continue
        if strategy == "fixed":
            runs = [(0, len(tokens))]
        else:
            runs = []
            start = 0
            for pos in range(1, len(tokens)):
                if tokens[pos][1] != tokens[pos - 1][1]:
                    runs.append((start, pos))
                    start = pos
            runs.append((start, len(tokens)))
        ordinal = 0
        for run_start, run_end in runs:
            start = run_start
            while start < run_end:
                end = min(start + limit, run_end)
                segment = tokens[start:end]
                section = segment[0][1]
                text = " ".join(word for word, _ in segment)
                identity = f"{epub_sha}|{strategy}|{doc.source}|{start}|{end}|{limit}|{overlap}"
                chunk_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
                chunks.append(Chunk(chunk_id, strategy, doc.source, doc.title, section,
                                    ordinal, text, len(segment),
                                    len({item[1] for item in segment}) > 1,
                                    0 if start == run_start else min(overlap, end - start)))
                ordinal += 1
                if end == run_end:
                    break
                start = end - overlap
    return chunks
