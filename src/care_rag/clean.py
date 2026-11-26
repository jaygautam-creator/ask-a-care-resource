"""Turn raw HTML/PDF snapshots into clean, sectioned documents.

Both formats are first converted to Markdown, then one parser splits Markdown into
`Section(heading, text)`. Sections matter because the chunker never crosses them and the
heading travels with every chunk ("5. Application for maintenance" is a far better citation
anchor than "page 3").

* HTML -> trafilatura (drops menus/footers, keeps headings and tables).
* PDF  -> pymupdf4llm with OCR off (layout-aware: reading order, heading levels, tables).
* Text repairs, all deliberately conservative:
  - NFKC, de-hyphenation, spacing around punctuation ("full -scale" -> "full-scale");
  - `repair_split_words` for PDF kerning artefacts ("m aintenance", "Departm ent");
  - section numbers split by kerning in headings ("1 0 . Alteration" -> "10. Alteration").
* Documents whose converter found almost no headings get a fallback: short Title-Case lines
  that stand alone are promoted to headings.

Known limitation: the Hindi PIB PDF uses a legacy font encoding, so some matras come out wrong
(e.g. "परिचय" extracts as "पररचय"). No extractor fixes that without OCR; it is kept and reported.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field

import pymupdf
import pymupdf4llm
import trafilatura

from care_rag.config import PROCESSED_DIR, RAW_DIR
from care_rag.fetch import load_manifest
from care_rag.sources import Source, load_sources

# --------------------------------------------------------------------------------------- types


@dataclass
class Section:
    heading: str
    text: str


@dataclass
class Document:
    id: str
    title: str
    url: str
    category: str
    lang: str
    publisher: str
    sections: list[Section] = field(default_factory=list)

    @property
    def n_words(self) -> int:
        return sum(len(s.text.split()) for s in self.sections)


# ----------------------------------------------------------------------------- text utilities

ALPHA = re.compile(r"[A-Za-z]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"[​‎‏﻿]", "", text)  # zero-width / direction marks
    return text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def fix_spacing(text: str) -> str:
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)  # "pen-\nsion" -> "pension"
    text = re.sub(r"(?<=\w) -(?=\w)", "-", text)  # "full -scale" -> "full-scale"
    text = re.sub(r"(?<=\S) +([,.;:)\]%])", r"\1", text)  # "word ," -> "word,"
    text = re.sub(r"([(\[]) +", r"\1", text)  # "( Legislative" -> "(Legislative"
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def repair_split_words(
    text: str, vocab: Counter, ref: Counter | None = None, max_parts: int = 4
) -> str:
    """Re-join words that PDF extraction split apart ("I ndia", "com m encem ent").

    `vocab` counts words in the whole corpus; `ref` counts words only in text that never went
    through PDF extraction (the HTML sources), so fragments like "ay" (from "m ay") are rare
    there. Adjacent alphabetic fragments separated by single spaces are merged when:
      (a) the merged word occurs in the corpus at least twice, and
      (b) it is more frequent than every multi-letter fragment is in `ref`; this keeps real
          phrases intact: "for m" never becomes "form", "in a" never becomes "ina", and
      (c) the merge does not strand the next fragment ("who m ay" must become "who may",
          not "whom ay").
    """
    ref = vocab if ref is None else ref
    tokens = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)

    def merge_len(i: int) -> int:
        for n in range(max_parts, 1, -1):
            idx = range(i, i + 2 * n - 1)
            if idx[-1] >= len(tokens):
                continue
            parts, seps = tokens[i : i + 2 * n - 1 : 2], tokens[i + 1 : i + 2 * n - 1 : 2]
            if any(s != " " for s in seps) or not all(ALPHA.fullmatch(p) for p in parts):
                continue
            merged = "".join(parts).lower()
            count = vocab[merged]
            if count < 2 or any(count <= ref[p.lower()] for p in parts if len(p) > 1):
                continue
            nxt = i + 2 * n - 1  # rule (c)
            if (
                nxt + 1 < len(tokens)
                and tokens[nxt] == " "
                and ALPHA.fullmatch(tokens[nxt + 1])
                and 1 < len(tokens[nxt + 1]) <= 3  # orphans are short: "ay", "ent"
                and ref[tokens[nxt + 1].lower()] < 2
            ):
                continue
            return n
        return 0

    out, i = [], 0
    while i < len(tokens):
        n = merge_len(i) if ALPHA.fullmatch(tokens[i]) else 0
        if n:
            out.append("".join(tokens[i : i + 2 * n - 1 : 2]))
            i += 2 * n - 1
        else:
            out.append(tokens[i])
            i += 1
    return "".join(out)


def fix_heading_numbers(heading: str) -> str:
    """'1 0 . Alteration' -> '10. Alteration'; 'Act, 2 0 0 7' -> 'Act, 2007'."""
    heading = re.sub(r"^(\d(?: \d){0,2}) ?\. ?", lambda m: m[1].replace(" ", "") + ". ", heading)
    return re.sub(r"\b([12]) ?([09]) ?(\d) ?(\d)\b", r"\1\2\3\4", heading)


# --------------------------------------------------------------------------------- converters


def html_to_markdown(html: str) -> str:
    return (
        trafilatura.extract(
            html,
            output_format="markdown",
            include_tables=True,
            include_links=False,
            include_images=False,
            favor_recall=True,
        )
        or ""
    )


def _plain_pdf_text(doc: pymupdf.Document) -> str:
    """PyMuPDF's own block reader: one paragraph per text block, blank line between blocks."""
    paras = []
    for page in doc:
        for block in page.get_text("blocks"):
            if block[6] == 0:  # text block (not image)
                paras.append(" ".join(block[4].split()))
    return "\n\n".join(p for p in paras if p)


def pdf_to_markdown(data: bytes) -> str:
    doc = pymupdf.open(stream=data, filetype="pdf")
    md = pymupdf4llm.to_markdown(
        doc, show_progress=False, use_ocr=False, header=False, footer=False
    )
    # Some PDFs (e.g. the PIB 2025 explainer) are laid out as "pictures with text"; there the
    # layout engine drops and reorders fragments, so we fall back to plain block extraction.
    if md.count("picture text") > 200:
        return _plain_pdf_text(doc)
    md = re.sub(r"<!-- (Start|End) of picture text -->", "", md)
    return md.replace("<br>", "\n")


# ------------------------------------------------------------------------- markdown -> sections

MD_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
BOLD_LINE = re.compile(r"^\*\*([^*]{3,120})\*\*:?$")
TABLE_RULE = re.compile(r"^\|?[\s|:-]+\|?$")
URL_ONLY = re.compile(r"^[>\s]*(<u>)?\s*\d*\s*https?://\S+\s*(</u>)?.*$")
SMALL_WORDS = {"of", "for", "and", "the", "in", "to", "a", "an", "on", "by", "with", "at", "or"}


def _strip_inline(line: str) -> str:
    line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)  # images
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)  # links -> text
    line = re.sub(r"</?(u|mark|sup|sub|b|i|strong|em|span)[^>]*>", "", line)  # inline html
    line = line.replace("**", "").replace("__", "")
    line = re.sub(r"^>\s?", "", line)  # blockquote
    line = re.sub(r"^\s*[-*+❖•●▪➢]\s+", "- ", line)  # unify bullets
    return line.rstrip()


def _looks_like_heading(line: str) -> bool:
    """Fallback for PDFs whose headings were not detected: short, Title-Case, no full stop."""
    words = line.split()
    if not (2 <= len(words) <= 9) or line.endswith((".", ",", ";", ":")) or line.startswith("-"):
        return False
    if any(ch.isdigit() for ch in words[0]) and len(words) < 3:
        return False
    caps = [w[0].isupper() for w in words if w.lower() not in SMALL_WORDS and w[0].isalpha()]
    return bool(caps) and all(caps) and words[-1].lower() not in SMALL_WORDS


def markdown_to_sections(md: str, fallback_headings: bool = False) -> list[Section]:
    sections: list[Section] = []
    heading, buf = "", []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(heading, body))

    lines = normalize(md).splitlines()
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if TABLE_RULE.fullmatch(line) or URL_ONLY.match(line):
            continue
        m = MD_HEADING.match(line) or BOLD_LINE.match(line)
        promoted = (
            not m
            and fallback_headings
            and (idx == 0 or not lines[idx - 1].strip())  # starts a paragraph
            and _looks_like_heading(_strip_inline(line))
        )
        if m or promoted:
            flush()
            heading = _strip_inline(m.group(1) if m else line).strip(" :#")
            buf = []
            continue
        buf.append(_strip_inline(line))
    flush()
    return sections


# ------------------------------------------------------------------------------------- driver


def _merge_tiny_sections(sections: list[Section], min_words: int = 25) -> list[Section]:
    """Fold sections with very little text into the next section.

    A tiny section's heading is kept as a line of text, so no information is lost; the merged
    section keeps the bigger section's heading (or the tiny one's, if the bigger has none).
    """
    out: list[Section] = []
    pending: list[Section] = []

    def as_text(sec: Section) -> str:
        return f"{sec.heading}\n{sec.text}".strip() if sec.heading else sec.text

    for sec in sections:
        if len(sec.text.split()) < min_words:
            pending.append(sec)
            continue
        if pending:
            heading = sec.heading or pending[-1].heading
            prefix = "\n\n".join(as_text(p) for p in pending)
            sec = Section(heading, f"{prefix}\n\n{sec.text}")
            pending = []
        out.append(sec)
    if pending:  # tiny sections at the very end go into the last real section
        tail = "\n\n".join(as_text(p) for p in pending)
        if out:
            out[-1] = Section(out[-1].heading, f"{out[-1].text}\n\n{tail}")
        else:
            out.append(Section(pending[0].heading, tail))
    return out


def parse_raw(data: bytes, kind: str) -> list[Section]:
    if kind == "pdf":
        md = pdf_to_markdown(data)
    else:
        md = html_to_markdown(data.decode("utf-8", "replace"))
    sections = markdown_to_sections(md)
    if kind == "pdf" and len(sections) < 3:  # converter missed the headings; use the fallback
        sections = markdown_to_sections(md, fallback_headings=True)
    return sections


def build_document(
    src: Source, sections: list[Section], vocab: Counter, ref: Counter
) -> Document:
    cleaned = []
    for s in sections:
        heading = fix_heading_numbers(fix_spacing(repair_split_words(s.heading, vocab, ref)))
        text = fix_spacing(repair_split_words(s.text, vocab, ref))
        if text:
            cleaned.append(Section(heading, text))
    return Document(
        src.id, src.title, src.url, src.category, src.lang, src.publisher,
        _merge_tiny_sections(cleaned),
    )


def clean_all() -> list[Document]:
    manifest = load_manifest()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parsed: list[tuple[Source, str, list[Section]]] = []
    for src in load_sources():
        if src.id not in manifest:
            print(f"  ! no snapshot for {src.id}; run fetch first")
            continue
        fname = manifest[src.id]["file"]
        kind = "pdf" if fname.endswith(".pdf") else "html"
        parsed.append((src, kind, parse_raw((RAW_DIR / fname).read_bytes(), kind)))

    def count_words(items) -> Counter:
        # URLs are removed first, otherwise "com" (from ".com") would look like a real word.
        return Counter(
            w.lower()
            for _, _, secs in items
            for s in secs
            for w in ALPHA.findall(re.sub(r"\S*(https?://|www\.)\S*", " ", s.text))
        )

    # A word split in one PDF ("I ndia") is usually intact somewhere else in the corpus.
    vocab = count_words(parsed)  # whole corpus: is the merged word real?
    ref = count_words([p for p in parsed if p[1] == "html"])  # clean text: is a fragment real?
    docs = [build_document(src, secs, vocab, ref) for src, _, secs in parsed]
    with open(PROCESSED_DIR / "documents.jsonl", "w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    return docs


def load_documents() -> list[Document]:
    docs = []
    with open(PROCESSED_DIR / "documents.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            d["sections"] = [Section(**s) for s in d["sections"]]
            docs.append(Document(**d))
    return docs


if __name__ == "__main__":
    for d in clean_all():
        print(f"{d.id:38} {len(d.sections):3d} sections {d.n_words:6d} words")
