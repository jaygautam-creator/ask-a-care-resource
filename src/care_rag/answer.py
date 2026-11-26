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
# e5 similarities are compressed (~0.70-0.90), so this sits low; see eval/calibrate_gate.py.
GATE_THRESHOLD = float(os.getenv("GATE_THRESHOLD", "0.78"))
CITE_RE = re.compile(r"\[(\d{1,2})\]")


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

    @property
    def uncited(self) -> bool:
        return not self.refused and not self.citations


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

    cited = sorted({int(n) for n in CITE_RE.findall(text)})
    valid = [n for n in cited if 1 <= n <= len(hits)]
    invalid = [n for n in cited if n not in valid]
    citations = [
        Citation(n, h.chunk.chunk_id, h.chunk.doc_id, h.chunk.title, h.chunk.section, h.chunk.url)
        for n in valid
        for h in [hits[n - 1]]
    ]
    return AnswerResult(question, text, False, None, citations, invalid, hits, search_query,
                        resp.provider, resp.model, lat)
