"""Structure-aware chunking.

Why this design (also in the README):
* Chunks never cross a section boundary, so every chunk has one meaningful heading to cite
  ("MWPSC Act > 5. Application for maintenance") and never mixes two schemes.
* Inside a section we pack whole paragraphs / list items / sentences up to ~300 words
  (~400 tokens), with ~50 words of sentence-aligned overlap so a fact split across a
  boundary is still retrievable from at least one chunk.
* The text that gets embedded is prefixed with "title | section", so even a short chunk such
  as "Rs 2,500 per month" carries the context (which scheme, which state) needed to match it.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from care_rag.clean import Document
from care_rag.config import ChunkConfig

# Sentence boundary: end punctuation (incl. Devanagari danda) followed by a likely sentence start.
# Abbreviations common in this corpus ("Rs.", "No.", "i.e.", "Govt.") must not end a sentence.
_ABBREV = (
    r"(?<!\bRs\.)(?<!\bNo\.)(?<!\bi\.e\.)(?<!\be\.g\.)(?<!\bGovt\.)(?<!\bDr\.)(?<!\bSr\.)"
    r"(?<!\bSh\.)(?<!\bSmt\.)(?<!\bviz\.)"
)
SENT_SPLIT = re.compile(r"(?<=[.!?।])" + _ABBREV + r"\s+(?=[\"'(\[]?[A-Z0-9ऀ-ॿ])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    section: str
    text: str
    lang: str
    category: str

    @property
    def embed_text(self) -> str:
        head = f"{self.title} | {self.section}" if self.section else self.title
        return f"{head}\n{self.text}"

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def split_units(text: str, max_words: int) -> list[str]:
    """Split section text into packable units: paragraphs, then sentences, then word windows."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n|\n(?=- )", text):
        para = para.strip()
        if not para:
            continue
        if len(para.split()) <= max_words:
            units.append(para)
            continue
        for sent in SENT_SPLIT.split(para):
            words = sent.split()
            if len(words) <= max_words:
                units.append(sent.strip())
            else:  # a giant "sentence" (tables, run-on lists): hard split
                units.extend(
                    " ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)
                )
    return units


def chunk_section(units: list[str], cfg: ChunkConfig) -> list[str]:
    """Greedily pack units up to target_words, with sentence-aligned overlap between chunks."""
    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0  # words in `current`
    n_carried = 0  # how many leading units of `current` are overlap copied from the last chunk
    for unit in units:
        n = len(unit.split())
        if current and size + n > cfg.target_words:
            chunks.append(current)
            carry: list[str] = []
            carried_words = 0
            for prev in reversed(current):  # last units that fit in the overlap budget
                w = len(prev.split())
                if carried_words + w > cfg.overlap_words:
                    break
                carry.insert(0, prev)
                carried_words += w
            current, size, n_carried = carry, carried_words, len(carry)
        current.append(unit)
        size += n
    if current:
        new_units = current[n_carried:]
        new_words = sum(len(u.split()) for u in new_units)
        if chunks and new_words < cfg.min_words:
            chunks[-1].extend(new_units)  # short tail: append to the previous chunk
        else:
            chunks.append(current)
    return ["\n".join(c) for c in chunks]


def chunk_documents(docs: list[Document], cfg: ChunkConfig) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        idx = 0
        for sec in doc.sections:
            for text in chunk_section(split_units(sec.text, cfg.target_words), cfg):
                out.append(
                    Chunk(
                        chunk_id=f"{doc.id}#{idx:03d}",
                        doc_id=doc.id,
                        title=doc.title,
                        url=doc.url,
                        section=sec.heading,
                        text=text,
                        lang=doc.lang,
                        category=doc.category,
                    )
                )
                idx += 1
    return out


def save_chunks(chunks: list[Chunk], path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path) -> list[Chunk]:
    with open(path, encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh]
