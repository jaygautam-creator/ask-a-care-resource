"""Answer a question: (optional rewrite) -> retrieve -> refusal gate -> LLM -> citation check.

Refusal has two layers:
1. A retrieval gate: if even the best passage is far from the question (low cosine similarity),
   refuse without calling the LLM. Cheap, and stops the LLM from "helpfully" answering
   off-topic questions from loosely related text. The threshold is calibrated on the eval set.
2. The prompt: for on-topic questions the corpus does not cover, the LLM must reply with the
   exact refusal sentence.

Citations are then validated: every [n] must refer to a passage that was actually shown.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from care_rag.config import CONFIGS, DEFAULT_CONFIG, REFUSAL, RetrievalConfig
from care_rag.llm import get_llm
from care_rag.prompts import ANSWER_SYSTEM, ANSWER_TEMPLATE, REWRITE_SYSTEM
from care_rag.retrieve import Hit, retrieve

# Cosine similarity (multilingual-e5) below which we refuse without asking the LLM.
# e5 similarities are compressed (~0.70-0.90), so this sits low. Calibrated by
# `python -m care_rag.cli eval` (results/gate_calibration): on our eval set unanswerable
# questions score 0.80-0.86, overlapping answerable ones (min 0.79), so the gate only
# catches clearly off-topic questions and the LLM prompt does the real refusal work.
GATE_THRESHOLD = float(os.getenv("GATE_THRESHOLD", "0.78"))
# Matches [2], [1][3] and the list form [1, 3] that models also produce.
CITE_RE = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]")


def cited_numbers(text: str) -> list[int]:
    return sorted({int(n) for group in CITE_RE.findall(text) for n in group.split(",")})


@dataclass
class Citation:
    n: int
    chunk_id: str
    doc_id: str
    title: str
    section: str
    url: str


@dataclass
class AnswerResult:
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None  # "gate" | "llm" | None
    citations: list[Citation]
    invalid_citations: list[int]  # [n] markers pointing at passages that were not shown
    hits: list[Hit]
    search_query: str
    provider: str | None = None
    model: str | None = None
    latency_ms: dict = field(default_factory=dict)
    unverified_numbers: list[str] = field(default_factory=list)  # numbers not in any passage

    @property
    def uncited(self) -> bool:
        return not self.refused and not self.citations


# A number, optionally followed by the unit that gives it meaning ("5 lakh" != "5 years").
NUM_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(lakh|crore|%|per cent)?", re.I)


def _norm_numbers(text: str) -> str:
    t = text.lower().replace("₹", " ").replace("lakhs", "lakh").replace("per cent", "%")
    t = re.sub(r"(?<=\d),(?=\d)", "", t)  # 1,00,000 -> 100000
    t = re.sub(r"(\d+\.\d*?)0+\b", r"\1", t)  # 8.20 -> 8.2
    t = re.sub(r"(\d)\.(?!\d)", r"\1", t)
    return re.sub(r"(\d)\s+(lakh|crore|%)", r"\1 \2", t)


def unverified_numbers(answer_text: str, hits: list[Hit]) -> list[str]:
    """Numbers in the answer that appear in none of the passages shown to the model.

    A deterministic check, independent of any LLM: amounts, ages, limits and phone numbers are
    exactly the facts a family acts on, and a fabricated "7 lakh" reads as convincingly as a
    real "5 lakh". Citation markers are ignored. A flag means "could not verify", not
    necessarily "wrong" (a source may spell a number out in words)."""
    body = _norm_numbers(CITE_RE.sub(" ", answer_text))
    source = _norm_numbers(" ".join(h.chunk.text for h in hits))
    missing = []
    for m in NUM_RE.finditer(body):
        num, unit = m.group(1).rstrip(".").replace(",", ""), (m.group(2) or "").lower()
        token = f"{num} {unit}" if unit else num
        tail = rf"\s*{re.escape(unit)}" if unit else r"(?![\d])"
        pattern = rf"(?<![\d.]){re.escape(num)}" + tail
        if not re.search(pattern, source) and token not in missing:
            missing.append(token)
    return missing


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for n, h in enumerate(hits, 1):
        c = h.chunk
        where = f"{c.title} - {c.section}" if c.section else c.title
        blocks.append(f"[{n}] ({where})\n{c.text}")
    return "\n\n".join(blocks)


def is_refusal(text: str) -> bool:
    t = text.strip().lower().replace("’", "'")
    return t.startswith(REFUSAL.lower().rstrip(".")) or t == REFUSAL.lower()


def rewrite_query(question: str) -> str:
    return get_llm("answer").generate(REWRITE_SYSTEM, question).text.strip().strip('"')


def answer(
    question: str,
    cfg: RetrievalConfig | None = None,
    gate: float | None = GATE_THRESHOLD,
    rewrite: bool = False,
) -> AnswerResult:
    cfg = cfg or CONFIGS[DEFAULT_CONFIG]
    lat: dict[str, float] = {}
    t0 = time.perf_counter()

    search_query = question
    if rewrite:
        search_query = f"{question}\n{rewrite_query(question)}"
        lat["rewrite"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    hits = retrieve(search_query, cfg)
    lat["retrieve"] = (time.perf_counter() - t1) * 1000

    best_sim = max((h.dense_sim or 0.0) for h in hits) if hits else 0.0
    if not hits or (gate is not None and best_sim < gate):
        lat["total"] = (time.perf_counter() - t0) * 1000
        return AnswerResult(question, REFUSAL, True, "gate", [], [], hits, search_query,
                            latency_ms=lat)

    prompt = ANSWER_TEMPLATE.format(
        context=format_context(hits), question=question, refusal=REFUSAL
    )
    resp = get_llm("answer").generate(ANSWER_SYSTEM, prompt)
    lat["generate"] = resp.latency_ms
    lat["total"] = (time.perf_counter() - t0) * 1000

    text = resp.text.strip()
    if is_refusal(text):
        return AnswerResult(question, REFUSAL, True, "llm", [], [], hits, search_query,
                            resp.provider, resp.model, lat)

    cited = cited_numbers(text)
    valid = [n for n in cited if 1 <= n <= len(hits)]
    invalid = [n for n in cited if n not in valid]
    citations = [
        Citation(n, h.chunk.chunk_id, h.chunk.doc_id, h.chunk.title, h.chunk.section, h.chunk.url)
        for n in valid
        for h in [hits[n - 1]]
    ]
    return AnswerResult(question, text, False, None, citations, invalid, hits, search_query,
                        resp.provider, resp.model, lat, unverified_numbers(text, hits))
