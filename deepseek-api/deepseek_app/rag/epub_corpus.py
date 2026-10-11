"""Read the ordered XHTML body of an EPUB without third-party dependencies."""

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile


EXCLUDED = {"OEBPS/cover.xhtml", "OEBPS/title.xhtml", "OEBPS/nav.xhtml",
            "OEBPS/README.xhtml", "OEBPS/rights.xhtml"}
OPF = "{http://www.idpf.org/2007/opf}"


@dataclass(frozen=True)
class Block:
    kind: str
    text: str
    level: int = 0


@dataclass(frozen=True)
class Document:
    source: str
    title: str
    blocks: tuple[Block, ...]


def _local(tag):
    return tag.rsplit("}", 1)[-1].lower()


def _text(element):
    return re.sub(r"\s+", " ", " ".join(element.itertext())).strip()


def _body_blocks(element):
    tag = _local(element.tag)
    if tag in {"nav", "script", "style", "head", "svg"}:
        return
    if tag in {"h1", "h2", "h3"}:
        value = _text(element)
        if value:
            yield Block("heading", value, int(tag[1]))
        return
    if tag in {"p", "pre", "li"}:
        # A list item can contain nested lists; emit the current item once,
        # then visit children that are themselves list containers.
        if tag == "li":
            parts = [element.text or ""]
            for child in element:
                if _local(child.tag) not in {"ul", "ol"}:
                    parts.extend(child.itertext())
                parts.append(child.tail or "")
            value = re.sub(r"\s+", " ", " ".join(parts)).strip()
        else:
            value = _text(element)
        if value:
            yield Block("body", value)
        if tag == "li":
            for child in element:
                if _local(child.tag) in {"ul", "ol"}:
                    yield from _body_blocks(child)
        return
    for child in element:
        yield from _body_blocks(child)


def read_epub(path: Path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"EPUB не найден: {path}")
    try:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        with ZipFile(path) as archive:
            container = ET.fromstring(archive.read("META-INF/container.xml"))
            rootfile = next(node.attrib["full-path"] for node in container.iter()
                            if _local(node.tag) == "rootfile")
            package = ET.fromstring(archive.read(rootfile))
            manifest = {item.attrib["id"]: item for item in package.findall(f"{OPF}manifest/{OPF}item")}
            opf_dir = PurePosixPath(rootfile).parent
            docs = []
            for ref in package.findall(f"{OPF}spine/{OPF}itemref"):
                item = manifest[ref.attrib["idref"]]
                if item.attrib.get("media-type") != "application/xhtml+xml" or "nav" in item.attrib.get("properties", "").split():
                    continue
                internal = str(opf_dir / item.attrib["href"].split("#", 1)[0])
                if internal in EXCLUDED:
                    continue
                tree = ET.fromstring(archive.read(internal))
                body = next((node for node in tree.iter() if _local(node.tag) == "body"), None)
                if body is None:
                    continue
                blocks = tuple(_body_blocks(body))
                if not any(block.kind == "body" for block in blocks):
                    continue
                h1 = next((block.text for block in blocks if block.kind == "heading" and block.level == 1), None)
                head_title = next((_text(node) for node in tree.iter() if _local(node.tag) == "title"), None)
                docs.append(Document(f"{path.name}:{internal}", h1 or head_title or PurePosixPath(internal).stem, blocks))
    except (BadZipFile, KeyError, ET.ParseError, StopIteration, OSError) as exc:
        raise ValueError(f"EPUB повреждён или не поддерживается: {path}: {exc}") from exc
    if not docs:
        raise ValueError("EPUB не содержит содержательных XHTML-страниц")
    return docs, digest
