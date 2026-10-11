import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from zipfile import ZipFile
from xml.etree import ElementTree as ET


from deepseek_app.rag.epub_corpus import read_epub, _body_blocks
from deepseek_app.rag.index_chunking import make_chunks
from deepseek_app.rag.document_index import build_index, load_index, search_index, IndexErrorMessage


def small_epub(path):
    def page(title, body):
        return f'''<html xmlns="http://www.w3.org/1999/xhtml"><head><title>{title}</title></head><body><section><h1>{title}</h1>{body}</section></body></html>'''

    opf = '''<package xmlns="http://www.idpf.org/2007/opf"><manifest>
      <item id="z" href="z.xhtml" media-type="application/xhtml+xml"/>
      <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
      <item id="a" href="a.xhtml" media-type="application/xhtml+xml"/>
      </manifest><spine><itemref idref="z"/><itemref idref="nav"/><itemref idref="a"/></spine></package>'''
    with ZipFile(path, "w") as z:
        z.writestr("META-INF/container.xml", '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>')
        z.writestr("OEBPS/content.opf", opf)
        z.writestr("OEBPS/nav.xhtml", page("Nav", "<p>skip me</p>"))
        z.writestr("OEBPS/z.xhtml", page("Zed", "<h2>First</h2><p>one two three</p><pre><code>print(42)</code></pre><nav><p>skip navigation</p></nav><h2>Second</h2><p>four five six</p>"))
        z.writestr("OEBPS/a.xhtml", page("Aye", "<p>seven eight nine</p>"))


class DocumentIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tmp_path = Path(self.temp.name)

    def test_epub_uses_spine_and_extracts_content(self):
        epub = self.tmp_path / "book.epub"
        small_epub(epub)
        docs, digest = read_epub(epub)
        self.assertEqual([d.source for d in docs], ["book.epub:OEBPS/z.xhtml", "book.epub:OEBPS/a.xhtml"])
        self.assertEqual(len(digest), 64)
        self.assertEqual([b.text for b in docs[0].blocks if b.kind == "body"], ["one two three", "print(42)", "four five six"])
        self.assertEqual([b.text for b in docs[0].blocks if b.kind == "heading"], ["Zed", "First", "Second"])

    def test_nested_list_and_inline_tail_are_not_duplicated(self):
        element = ET.fromstring("<ul><li>alpha <em>beta</em> tail<ul><li>nested</li></ul></li></ul>")
        self.assertEqual([b.text for b in _body_blocks(element)], ["alpha beta tail", "nested"])

    def test_chunking_covers_words_preserves_sections_and_stable_ids(self):
        epub = self.tmp_path / "book.epub"
        small_epub(epub)
        docs, digest = read_epub(epub)
        fixed = make_chunks(docs, digest, "fixed", limit=4, overlap=1)
        structural = make_chunks(docs, digest, "structural", limit=4, overlap=1)
        self.assertTrue(any(c.crosses_section for c in fixed))
        self.assertFalse(any(c.crosses_section for c in structural))
        self.assertEqual({c.source for c in fixed}, {c.source for c in structural})
        for strategy, chunks in (("fixed", fixed), ("structural", structural)):
            self.assertEqual([c.chunk_id for c in chunks], [c.chunk_id for c in make_chunks(docs, digest, strategy, 4, 1)])
            self.assertTrue(all(c.source and c.title and c.section and c.text and c.chunk_id for c in chunks))
            for doc in docs:
                expected = [w for b in doc.blocks if b.kind == "body" for w in b.text.split()]
                observed = [w for c in chunks if c.source == doc.source for w in c.text.split()]
                self.assertTrue(all(word in observed for word in expected))

    def test_build_reopen_search_and_atomic_failure(self):
        epub = self.tmp_path / "book.epub"
        db = self.tmp_path / "index.sqlite3"
        small_epub(epub)

        class Embedder:
            model = "fake"

            def embed(self, texts):
                return [[float("print" in text), float("seven" in text), 1.0] for text in texts]

        build_index(epub, db, Embedder(), limit=4, overlap=1, batch_size=2)
        meta, rows = load_index(db)
        self.assertEqual(meta["model"], "fake")
        self.assertEqual(len({r["strategy"] for r in rows}), 2)
        self.assertTrue(all(len(json.loads(r["vector"])) == 3 for r in rows))
        self.assertTrue(search_index(db, "print", "structural", Embedder())[0]["source"].endswith("z.xhtml"))
        original = db.read_bytes()

        class BadEmbedder(Embedder):
            def embed(self, texts):
                return [[float("nan"), 0, 1] for _ in texts]

        with self.assertRaises(IndexErrorMessage):
            build_index(epub, db, BadEmbedder(), limit=4, overlap=1)
        self.assertEqual(db.read_bytes(), original)

    def test_missing_and_invalid_index(self):
        with self.assertRaisesRegex(IndexErrorMessage, "индекс"):
            load_index(self.tmp_path / "missing.sqlite3")
        with sqlite3.connect(self.tmp_path / "invalid.sqlite3") as db:
            db.execute("create table unrelated(x)")
        with self.assertRaisesRegex(IndexErrorMessage, "схем"):
            load_index(self.tmp_path / "invalid.sqlite3")
