"""Answer a question: (optional rewrite) -> retrieve -> refusal gate -> LLM -> citation check.

Refusal has two layers:
1. A retrieval gate: if even the best passage is far from the question (low cosine similarity),
   refuse without calling the LLM. Cheap, and stops the LLM from "helpfully" answering
   off-topic questions from loosely related text. The threshold is calibrated on the eval set.
2. The prompt: for on-topic questions the corpus does not cover, the LLM must reply with the
   exact refusal sentence.

Citations are then validated: every [n] must refer to a passage that was actually shown.

Verify and repair: deterministic checks run on every draft answer (every number must appear in
the passages; a Hindi question must get a Devanagari answer). A flagged draft is sent back to
the model once with the problems listed, and the retry is kept only if it is strictly better.
Whatever is still unverified is returned to the caller, so the UI can warn about it.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from care_rag.config import CONFIGS, DEFAULT_CONFIG, REFUSAL, RetrievalConfig
from care_rag.llm import get_llm
from care_rag.prompts import ANSWER_SYSTEM, ANSWER_TEMPLATE, REPAIR_TEMPLATE, REWRITE_SYSTEM
from care_rag.retrieve import Hit, retrieve_multi

# Cosine similarity (multilingual-e5) below which we refuse without asking the LLM.
# e5 similarities are compressed (~0.70-0.90), so this sits low. Calibrated by
# `python -m care_rag.cli eval` (results/gate_calibration): on our eval set unanswerable
# questions score 0.80-0.86, overlapping answerable ones (min 0.79), so the gate only
# catches clearly off-topic questions and the LLM prompt does the real refusal work.
GATE_THRESHOLD = float(os.getenv("GATE_THRESHOLD", "0.78"))
# ANSWER_REPAIR=0 turns the verify-and-repair retry off (for comparison runs).
REPAIR = os.getenv("ANSWER_REPAIR", "1") != "0"
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
    cached: bool = False  # the LLM response came from the local cache
    draft_issues: list[str] = field(default_factory=list)  # what the checks flagged in the draft
    repaired: bool = False  # the draft was replaced by a corrected retry

    @property
    def uncited(self) -> bool:
        return not self.refused and not self.citations


# A number, optionally followed by the unit that gives it meaning ("5 lakh" != "5 years").
NUM_RE = re.compile(r"(?<!\w)(?<!\d\.)(\d[\d,]*(?:\.\d+)?)\s*(lakh|crore|%|per cent)?", re.I)

_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
          "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
WORD_NUM = {w: i for i, w in enumerate(_UNITS)} | {w: 20 + 10 * i for i, w in enumerate(_TENS)}
WORD_RE = re.compile(
    r"\b(?:(?:" + "|".join(WORD_NUM) + r"|hundred|thousand)(?:[\s-]+(?:and[\s-]+)?|\b))+",
    re.I,
)


def _words_to_digits(text: str) -> str:
    """"ninety days" -> "90 days", "ten thousand rupees" -> "10000 rupees" (the Act spells
    numbers out, while answers use digits)."""

    def conv(m: re.Match) -> str:
        total, cur = 0, 0
        for w in re.findall(r"[a-z]+", m.group(0).lower()):
            if w in WORD_NUM:
                cur += WORD_NUM[w]
            elif w == "hundred":
                cur = max(cur, 1) * 100
            elif w == "thousand":
                total, cur = total + max(cur, 1) * 1000, 0
        return f"{total + cur} "

    return WORD_RE.sub(conv, text)


_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_HINDI_UNITS = (("लाख", "lakh"), ("करोड़", "crore"), ("करोड", "crore"), ("प्रतिशत", "%"))


def _norm_numbers(text: str) -> str:
    t = text.translate(_DEVANAGARI_DIGITS)
    for hi, en in _HINDI_UNITS:  # "5 लाख" and "5 lakh" are the same fact
        t = t.replace(hi, en)
    t = t.lower().replace("₹", " ").replace("lakhs", "lakh").replace("per cent", "%")
    t = re.sub(r"(?<=\d),(?=\d)", "", t)  # 1,00,000 -> 100000
    t = re.sub(r"(\d+\.\d*?)0+\b", r"\1", t)  # 8.20 -> 8.2
    t = re.sub(r"(\d)\.(?!\d)", r"\1", t)
    return re.sub(r"(\d)\s+(lakh|crore|%)", r"\1 \2", t)


def unverified_numbers(answer_text: str, hits: list[Hit], question: str = "") -> list[str]:
    """Numbers in the answer that appear neither in the passages shown to the model (including
    their title and section, which the model also sees) nor in the user's own question.

    A deterministic check, independent of any LLM: amounts, ages, limits and phone numbers are
    exactly the facts a family acts on, and a fabricated "7 lakh" reads as convincingly as a
    real "5 lakh". Citation markers are ignored; numbers written as words in the source count.
    A flag means "could not verify", not necessarily "wrong"."""
    body = _norm_numbers(CITE_RE.sub(" ", answer_text))
    source = _norm_numbers(_words_to_digits(f"{question}\n{format_context(hits)}"))
    source = CITE_RE.sub(" ", source)  # passage numbers "[3]" are not facts
    missing = []
    for m in NUM_RE.finditer(body):
        num, unit = m.group(1).rstrip(".").replace(",", ""), (m.group(2) or "").lower()
        token = f"{num} {unit}" if unit else num
        tail = rf"\s*{re.escape(unit)}" if unit else r"(?![\d])"
        pattern = rf"(?<!\d)(?<!\d\.){re.escape(num)}" + tail
        if not re.search(pattern, source) and token not in missing:
            missing.append(token)
    return missing


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")


def _devanagari_share(text: str) -> float:
    dev, lat = len(DEVANAGARI_RE.findall(text)), len(LATIN_RE.findall(text))
    return dev / (dev + lat) if dev + lat else 0.0


def script_mismatch(question: str, answer_text: str) -> bool:
    """A question written in Devanagari answered (mostly) in Roman script. Scheme names and
    citations are often Latin, so the answer only needs some real Devanagari text."""
    return _devanagari_share(question) > 0.5 and _devanagari_share(answer_text) < 0.3


def check_answer(question: str, answer_text: str, hits: list[Hit]) -> tuple[list[str], list[str]]:
    """(unverified numbers, human-readable issues) for a draft answer."""
    nums = unverified_numbers(answer_text, hits, question)
    issues = []
    if nums:
        issues.append("These numbers do not appear in the passages: " + ", ".join(nums)
                      + ". Remove them, or replace them with the exact figures a passage states.")
    if script_mismatch(question, answer_text):
        issues.append("The question is in Hindi (Devanagari script), but the draft is in Roman "
                      "script. Write the answer in Hindi, in Devanagari script.")
    return nums, issues


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
    return _rewrite(question)[0]


def _rewrite(question: str) -> tuple[str, float]:
    resp = get_llm("answer").generate(REWRITE_SYSTEM, question)
    return resp.text.strip().strip('"'), resp.latency_ms


def answer(
    question: str,
    cfg: RetrievalConfig | None = None,
    gate: float | None = GATE_THRESHOLD,
    rewrite: bool = False,
    repair: bool = REPAIR,
) -> AnswerResult:
    cfg = cfg or CONFIGS[DEFAULT_CONFIG]
    lat: dict[str, float] = {}

    queries = [question]
    if rewrite:
        rewritten, lat["rewrite"] = _rewrite(question)  # original call latency, even if cached
        queries.append(rewritten)  # searched separately, rankings fused (retrieve_multi)
    search_query = queries[-1]

    t1 = time.perf_counter()
    hits = retrieve_multi(queries, cfg)
    lat["retrieve"] = (time.perf_counter() - t1) * 1000

    best_sim = max((h.dense_sim or 0.0) for h in hits) if hits else 0.0
    if not hits or (gate is not None and best_sim < gate):
        lat["total"] = sum(lat.values())
        return AnswerResult(question, REFUSAL, True, "gate", [], [], hits, search_query,
                            latency_ms=lat)

    prompt = ANSWER_TEMPLATE.format(
        context=format_context(hits), question=question, refusal=REFUSAL
    )
    resp = get_llm("answer").generate(ANSWER_SYSTEM, prompt)
    lat["generate"] = resp.latency_ms
    # total = sum of stages; LLM stages use the original (uncached) call latency, so the
    # number means "what a user waits", even when the evaluation replays from the cache
    lat["total"] = sum(lat.values())

    text = resp.text.strip()
    if is_refusal(text):
        return AnswerResult(question, REFUSAL, True, "llm", [], [], hits, search_query,
                            resp.provider, resp.model, lat, cached=resp.cached)

    nums, issues = check_answer(question, text, hits)
    draft_issues, repaired = issues, False
    if issues and repair:
        fix = get_llm("answer").generate(ANSWER_SYSTEM, REPAIR_TEMPLATE.format(
            context=format_context(hits), question=question, draft=text,
            issues="\n".join(f"- {i}" for i in issues)))
        lat["repair"] = fix.latency_ms
        lat["total"] = sum(v for k, v in lat.items() if k != "total")
        new_text = fix.text.strip()
        new_nums, new_issues = check_answer(question, new_text, hits)
        if not is_refusal(new_text) and cited_numbers(new_text) and len(new_issues) <= len(
            issues
        ) and (len(new_nums), len(new_issues)) < (len(nums), len(issues)):
            text, nums, repaired = new_text, new_nums, True

    cited = cited_numbers(text)
    valid = [n for n in cited if 1 <= n <= len(hits)]
    invalid = [n for n in cited if n not in valid]
    citations = [
        Citation(n, h.chunk.chunk_id, h.chunk.doc_id, h.chunk.title, h.chunk.section, h.chunk.url)
        for n in valid
        for h in [hits[n - 1]]
    ]
    return AnswerResult(question, text, False, None, citations, invalid, hits, search_query,
                        resp.provider, resp.model, lat, nums, cached=resp.cached,
                        draft_issues=draft_issues, repaired=repaired)
