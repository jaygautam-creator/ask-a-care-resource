"""Command line: `python -m care_rag.cli <command>`.

  ingest      fetch (only missing files unless --refresh) -> clean -> chunk -> embed -> index
  ask         answer one question with citations
  eval        run the evaluation (retrieval for all configs; answers for the experiment configs)
  report      rebuild results/RESULTS.md from saved runs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")  # third-party deprecation noise; real errors still raise
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# The answer-level experiment: same retrieval stack, with and without each change.
ANSWER_RUNS = {
    "hybrid": ("hybrid", False),
    "hybrid_rerank_small": ("hybrid_rerank_small", False),
    "hybrid_rerank_small_rewrite": ("hybrid_rerank_small", True),
}


def cmd_ingest(args) -> None:
    from care_rag.clean import clean_all
    from care_rag.config import CONFIGS
    from care_rag.fetch import fetch_all
    from care_rag.index import build_index

    manifest = fetch_all(refresh=args.refresh)
    print(f"raw snapshot: {len(manifest)} sources")
    docs = clean_all()
    print(f"cleaned: {len(docs)} documents, "
          f"{sum(len(s.text.split()) for d in docs for s in d.sections):,} words")
    for chunk_cfg in sorted({c.chunk for c in CONFIGS.values()}, key=lambda c: -c.target_words):
        idx = build_index(chunk_cfg)
        print(f"index {chunk_cfg.name}: {len(idx.chunks)} chunks")


def cmd_ask(args) -> None:
    from care_rag.answer import answer
    from care_rag.config import CONFIGS, DEFAULT_CONFIG

    res = answer(args.question, CONFIGS[args.config or DEFAULT_CONFIG], rewrite=args.rewrite)
    print("\n" + res.answer + "\n")
    if res.citations:
        print("Sources:")
        for c in res.citations:
            print(f"  [{c.n}] {c.title} - {c.section or '(intro)'}\n      {c.url}")
    if res.invalid_citations:
        print(f"  warning: citations {res.invalid_citations} do not match any passage")
    why = {"gate": "no passage was close enough to the question",
           "llm": "the model found no answer in the retrieved passages"}
    if res.refused:
        print(f"(refused: {why[res.refusal_reason]})")
    lat = ", ".join(f"{k} {v / 1000:.1f}s" for k, v in res.latency_ms.items())
    print(f"\n[{res.provider or 'no LLM call'}:{res.model or ''}] {lat}")


def cmd_eval(args) -> None:
    from care_rag.config import CONFIGS, DEFAULT_CONFIG
    from care_rag.evaluate import (
        calibrate_gate,
        config_meta,
        eval_answers,
        eval_retrieval,
        load_eval_set,
        save_run,
    )
    from care_rag.report import write_report

    qs = load_eval_set()
    print(f"{len(qs)} questions ({sum(not q.answerable for q in qs)} unanswerable)")

    if not args.skip_retrieval:
        wanted = args.configs.split(",") if args.configs else list(CONFIGS) + ["rewrite"]
        for name, cfg in CONFIGS.items():
            if name not in wanted:
                continue
            print(f"retrieval: {name}")
            r = eval_retrieval(cfg, qs)
            save_run(f"retrieval_{name}", r, config_meta(cfg, kind="retrieval"))
            if name == DEFAULT_CONFIG:
                gate = calibrate_gate(r["rows"])
                save_run("gate_calibration", {"rows": [], "summary": gate},
                         config_meta(cfg, kind="gate"))
                print(f"  gate: recommended threshold {gate['recommended']}")
        if "rewrite" in wanted:
            print(f"retrieval: {DEFAULT_CONFIG} + query rewrite")
            r = eval_retrieval(CONFIGS[DEFAULT_CONFIG], qs, rewrite=True)
            save_run(f"retrieval_{DEFAULT_CONFIG}_rewrite", r,
                     config_meta(CONFIGS[DEFAULT_CONFIG], kind="retrieval", rewrite=True))

    if not args.retrieval_only:
        runs = args.runs.split(",") if args.runs else list(ANSWER_RUNS)
        for run in runs:
            cfg_name, rewrite = ANSWER_RUNS[run]
            print(f"answers: {run}")
            r = eval_answers(CONFIGS[cfg_name], qs, rewrite=rewrite, use_judge=not args.no_judge)
            save_run(f"answers_{run}", r, config_meta(CONFIGS[cfg_name], kind="answers",
                                                      rewrite=rewrite))
            print(json.dumps({k: v for k, v in r["summary"].items() if k != "by_style"}))
    print(f"report: {write_report()}")


def cmd_report(args) -> None:
    from care_rag.report import write_report

    print(write_report())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="care_rag", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="build the corpus and indexes")
    s.add_argument("--refresh", action="store_true", help="re-download every source")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("ask", help="ask a question")
    s.add_argument("question")
    s.add_argument("--config", help="retrieval config name (default: DEFAULT_CONFIG)")
    s.add_argument("--rewrite", action="store_true", help="LLM query rewrite before retrieval")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("eval", help="run the evaluation")
    s.add_argument("--retrieval-only", action="store_true", help="no LLM answer generation")
    s.add_argument("--skip-retrieval", action="store_true")
    s.add_argument("--configs", help="comma-separated retrieval configs (plus 'rewrite')")
    s.add_argument("--runs", help=f"comma-separated subset of {','.join(ANSWER_RUNS)}")
    s.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("report", help="rebuild results/RESULTS.md")
    s.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
