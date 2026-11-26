"""Evaluation: retrieval metrics, answer metrics (LLM judge + objective checks), gate calibration.

Scoring, in plain words (also in the README):

Retrieval (answerable questions only; documents are de-duplicated in rank order, so five chunks
of the same document count as one retrieved document):
* hit@k     - is ANY expected document among the top-k retrieved documents?
* recall@k  - fraction of the question's expected GROUPS covered in the top-k (a multi-source
              question with two groups scores 0.5 if only one part was found).
* MRR       - 1 / rank of the first expected document (0 if none in the top 10).

Answers:
* refusal accuracy   - unanswerable questions that were refused (higher is better).
* false refusal rate - answerable questions that were refused (lower is better).
* key-fact recall    - fraction of a question's key facts (numbers / names written in the eval
                       set) present in the answer, after normalising commas, case and "₹/Rs".
                       Objective and cheap, but only covers what we wrote down.
* citation validity  - every [n] in the answer points at a passage that was actually shown.
* cited-doc precision- fraction of cited passages that come from an expected document.
* faithfulness       - LLM judge (a different model family from the answerer) splits the
                       answer into atomic claims and marks each as supported or not by the
                       passages; score = supported / claims.
* citation correctness - judge: the passage numbers next to a claim actually contain it.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import asdict, dataclass, field, replace

import yaml

from care_rag.answer import AnswerResult, answer, format_context, rewrite_query
from care_rag.config import EVAL_DIR, RESULTS_DIR, RetrievalConfig
from care_rag.llm import get_llm, parse_json
from care_rag.prompts import JUDGE_SYSTEM, JUDGE_TEMPLATE
from care_rag.retrieve import retrieve

RETRIEVAL_DEPTH = 10  # documents considered for MRR / hit@k


@dataclass
class EvalQuestion:
    id: str
    style: str
    question: str
    answerable: bool
    expected: list[list[str]] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def relevant(self) -> set[str]:
        return {d for group in self.expected for d in group}


def load_eval_set(path=EVAL_DIR / "eval_set.yaml") -> list[EvalQuestion]:
    data = yaml.safe_load(open(path, encoding="utf-8"))
    qs = [EvalQuestion(**q) for q in data["questions"]]
    assert len({q.id for q in qs}) == len(qs), "duplicate question ids"
    return qs


# ------------------------------------------------------------------------------- retrieval


def ranked_docs(doc_ids: list[str]) -> list[str]:
    """Chunk-level ranking -> document ranking (first occurrence wins)."""
    seen: list[str] = []
    for d in doc_ids:
        if d not in seen:
            seen.append(d)
    return seen


def retrieval_scores(q: EvalQuestion, docs: list[str]) -> dict:
    rel = q.relevant
    first = next((i for i, d in enumerate(docs, 1) if d in rel), None)
    out = {"mrr": 1.0 / first if first else 0.0, "first_relevant_rank": first}
    for k in (1, 3, 5):
        top = set(docs[:k])
        out[f"hit@{k}"] = float(bool(top & rel))
        out[f"recall@{k}"] = sum(bool(top & set(g)) for g in q.expected) / len(q.expected)
    return out


def search_query(question: str, rewrite: bool) -> str:
    return f"{question}\n{rewrite_query(question)}" if rewrite else question


def eval_retrieval(cfg: RetrievalConfig, qs: list[EvalQuestion], rewrite: bool = False) -> dict:
    """Retrieval-only run (no answer generation). Also records the refusal-gate signal."""
    deep = replace(cfg, top_k=max(cfg.top_k, 20))  # deeper list, so ranks up to 10 docs exist
    rows = []
    for q in qs:
        sq = search_query(q.question, rewrite)
        t0 = time.perf_counter()
        hits = retrieve(sq, deep)
        ms = (time.perf_counter() - t0) * 1000
        docs = ranked_docs([h.chunk.doc_id for h in hits])[:RETRIEVAL_DEPTH]
        row = {
            "id": q.id,
            "style": q.style,
            "answerable": q.answerable,
            "retrieved_docs": docs,
            "best_sim": max(h.dense_sim for h in hits[: cfg.top_k]) if hits else 0.0,
            "retrieve_ms": ms,
        }
        if q.answerable:
            row.update(retrieval_scores(q, docs))
        rows.append(row)
    return {"rows": rows, "summary": summarize_retrieval(rows)}


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 3) if xs else None


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return round(xs[i])


def summarize_retrieval(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    s = {m: _mean(r[m] for r in ans) for m in
         ("hit@1", "hit@3", "hit@5", "recall@5", "mrr")}
    s["n"] = len(ans)
    lat = [r["retrieve_ms"] for r in rows]
    s["retrieve_ms_p50"], s["retrieve_ms_p95"] = pct(lat, 50), pct(lat, 95)
    s["by_style"] = {
        st: {"n": len(g), "hit@5": _mean(r["hit@5"] for r in g), "mrr": _mean(r["mrr"] for r in g)}
        for st in ("formal", "stressed", "hinglish", "hindi")
        if (g := [r for r in ans if r["style"] == st])
    }
    return s


# ---------------------------------------------------------------------------------- answers


def norm(text: str) -> str:
    t = text.lower().replace("₹", "rs ").replace(",", "").replace("lakhs", "lakh")
    return re.sub(r"\s+", " ", t)


def key_fact_recall(q: EvalQuestion, text: str) -> float | None:
    if not q.key_facts:
        return None
    t = norm(text)
    return sum(norm(f) in t for f in q.key_facts) / len(q.key_facts)


def judge(res: AnswerResult) -> dict:
    """LLM-as-judge over the passages the answerer saw. Returns per-claim verdicts."""
    prompt = JUDGE_TEMPLATE.format(
        context=format_context(res.hits), question=res.question, answer=res.answer
    )
    resp = get_llm("judge").generate(JUDGE_SYSTEM, prompt, json_mode=True)
    try:
        verdict = parse_json(resp.text)
    except (ValueError, json.JSONDecodeError):
        return {"error": "unparseable judge output", "judge": f"{resp.provider}:{resp.model}"}
    claims = verdict.get("claims") or []
    n = len(claims)
    return {
        "judge": f"{resp.provider}:{resp.model}",
        "n_claims": n,
        "faithfulness": sum(bool(c.get("supported")) for c in claims) / n if n else None,
        "citation_correct": sum(bool(c.get("citation_correct")) for c in claims) / n if n else None,
        "answers_question": verdict.get("answers_the_question"),
        "unsupported": [c.get("claim") for c in claims if not c.get("supported")],
    }


def eval_answers(
    cfg: RetrievalConfig,
    qs: list[EvalQuestion],
    rewrite: bool = False,
    gate: float | None = None,
    use_judge: bool = True,
) -> dict:
    from care_rag.answer import GATE_THRESHOLD

    gate = GATE_THRESHOLD if gate is None else gate
    rows = []
    for i, q in enumerate(qs, 1):
        res = answer(q.question, cfg, gate=gate, rewrite=rewrite)
        docs = ranked_docs([h.chunk.doc_id for h in res.hits])
        cited_docs = [c.doc_id for c in res.citations]
        row = {
            "id": q.id,
            "style": q.style,
            "answerable": q.answerable,
            "question": q.question,
            "answer": res.answer,
            "refused": res.refused,
            "refusal_reason": res.refusal_reason,
            "provider": f"{res.provider}:{res.model}" if res.provider else None,
            "retrieved_docs": docs,
            "cited_docs": cited_docs,
            "invalid_citations": res.invalid_citations,
            "best_sim": max((h.dense_sim or 0) for h in res.hits) if res.hits else 0.0,
            "latency_ms": res.latency_ms,
        }
        if q.answerable:
            row.update(retrieval_scores(q, docs))
            row["key_fact_recall"] = 0.0 if res.refused else key_fact_recall(q, res.answer)
            if not res.refused:
                row["citation_valid"] = float(not res.invalid_citations and bool(cited_docs))
                row["cited_doc_precision"] = (
                    sum(d in q.relevant for d in cited_docs) / len(cited_docs)
                    if cited_docs else 0.0
                )
        if use_judge and not res.refused:
            row.update(judge(res))
        rows.append(row)
        print(f"  [{i:2d}/{len(qs)}] {q.id} {'REFUSED' if res.refused else 'answered'} "
              f"({row['provider']})", flush=True)
    return {"rows": rows, "summary": summarize_answers(rows)}


def summarize_answers(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans if not r["refused"]]
    judged = [r for r in rows if r.get("faithfulness") is not None]
    gen = [r["latency_ms"]["generate"] for r in rows if "generate" in r["latency_ms"]
           and r["latency_ms"]["generate"] > 0]  # cached calls report 0
    total = [r["latency_ms"]["total"] for r in rows]
    providers: dict[str, int] = {}
    for r in rows:
        if r["provider"]:
            providers[r["provider"]] = providers.get(r["provider"], 0) + 1
    s = {
        "n": len(rows),
        "refusal_accuracy": _mean(float(r["refused"]) for r in una),
        "false_refusal_rate": _mean(float(r["refused"]) for r in ans),
        "key_fact_recall": _mean(r.get("key_fact_recall") for r in ans),
        "citation_valid": _mean(r.get("citation_valid") for r in answered),
        "cited_doc_precision": _mean(r.get("cited_doc_precision") for r in answered),
        "faithfulness": _mean(r["faithfulness"] for r in judged),
        "citation_correct": _mean(r.get("citation_correct") for r in judged),
        "hit@5": _mean(r["hit@5"] for r in ans),
        "mrr": _mean(r["mrr"] for r in ans),
        "generate_ms_p50": pct(gen, 50),
        "generate_ms_p95": pct(gen, 95),
        "total_ms_p50": pct(total, 50),
        "total_ms_p95": pct(total, 95),
        "providers": providers,
        "refused_by": {
            "gate": sum(r["refusal_reason"] == "gate" for r in rows),
            "llm": sum(r["refusal_reason"] == "llm" for r in rows),
        },
    }
    s["by_style"] = {
        st: {
            "n": len(g),
            "false_refusal_rate": _mean(float(r["refused"]) for r in g if r["answerable"]),
            "key_fact_recall": _mean(r.get("key_fact_recall") for r in g if r["answerable"]),
            "refusal_accuracy": _mean(float(r["refused"]) for r in g if not r["answerable"]),
        }
        for st in ("formal", "stressed", "hinglish", "hindi")
        if (g := [r for r in rows if r["style"] == st])
    }
    return s


# ------------------------------------------------------------------------- gate calibration


def calibrate_gate(rows: list[dict]) -> dict:
    """Sweep the similarity threshold on retrieval rows; report how each value would split
    answerable vs unanswerable questions. The gate is a cheap first layer: we want it to
    refuse *only* questions that are clearly off-corpus, so we pick the highest threshold
    that refuses no answerable question (false refusals are worse than an extra LLM call:
    the LLM is the second refusal layer)."""
    ans = sorted(r["best_sim"] for r in rows if r["answerable"])
    una = sorted(r["best_sim"] for r in rows if not r["answerable"])
    sweep = []
    for t in [round(0.70 + 0.005 * i, 3) for i in range(41)]:
        sweep.append({
            "threshold": t,
            "false_refusals": sum(s < t for s in ans),
            "unanswerable_refused": sum(s < t for s in una),
        })
    safe = [s for s in sweep if s["false_refusals"] == 0]
    return {
        "answerable_sims": [round(x, 3) for x in ans],
        "unanswerable_sims": [round(x, 3) for x in una],
        "recommended": safe[-1]["threshold"] if safe else None,
        "sweep": sweep,
    }


# ---------------------------------------------------------------------------------- storage


def save_run(name: str, result: dict, meta: dict) -> None:
    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_question.jsonl", "w", encoding="utf-8") as fh:
        for r in result["rows"]:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    (out / "summary.json").write_text(
        json.dumps({"meta": meta, "summary": result["summary"]}, indent=2, ensure_ascii=False)
    )


def load_summary(name: str) -> dict | None:
    p = RESULTS_DIR / name / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def config_meta(cfg: RetrievalConfig, **extra) -> dict:
    return {"config": asdict(cfg), **extra, "date": time.strftime("%Y-%m-%d %H:%M")}

