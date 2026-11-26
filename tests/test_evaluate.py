from care_rag.answer import is_refusal
from care_rag.evaluate import (
    EvalQuestion,
    calibrate_gate,
    key_fact_recall,
    load_eval_set,
    ranked_docs,
    retrieval_scores,
)


def q(expected, facts=()):
    return EvalQuestion("t", "formal", "?", True, expected, list(facts))


def test_ranked_docs_dedupes_in_order():
    assert ranked_docs(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_retrieval_scores_single_group():
    s = retrieval_scores(q([["x", "y"]]), ["a", "y", "b"])
    assert s["mrr"] == 0.5 and s["hit@1"] == 0.0 and s["hit@3"] == 1.0


def test_retrieval_scores_multi_group_partial():
    s = retrieval_scores(q([["x"], ["z"]]), ["x", "a", "b", "c", "d", "z"])
    assert s["recall@5"] == 0.5 and s["hit@5"] == 1.0 and s["mrr"] == 1.0


def test_retrieval_scores_miss():
    s = retrieval_scores(q([["x"]]), ["a", "b"])
    assert s["mrr"] == 0.0 and s["first_relevant_rank"] is None


def test_key_fact_normalisation():
    item = q([["x"]], ["10,000", "5 lakh"])
    assert key_fact_recall(item, "up to ₹10000 a month and Rs 5 lakhs cover") == 1.0
    assert key_fact_recall(item, "up to 10,000") == 0.5
    assert key_fact_recall(q([["x"]]), "anything") is None


def test_gate_recommends_threshold_without_false_refusals():
    rows = [{"answerable": True, "best_sim": s} for s in (0.80, 0.85, 0.9)]
    rows += [{"answerable": False, "best_sim": s} for s in (0.75, 0.79)]
    g = calibrate_gate(rows)
    assert g["recommended"] == 0.8
    best = next(x for x in g["sweep"] if x["threshold"] == 0.8)
    assert best["false_refusals"] == 0 and best["unanswerable_refused"] == 2


def test_is_refusal_variants():
    assert is_refusal("I don't know from these sources.")
    assert is_refusal("I don’t know from these sources")
    assert not is_refusal("The helpline is 14567 [1].")


def test_eval_set_meets_brief():
    qs = load_eval_set()
    assert len(qs) >= 25
    assert sum(not x.answerable for x in qs) >= 5
    assert sum(x.style in ("stressed", "hinglish", "hindi") for x in qs) >= 5
    for x in qs:
        assert x.answerable == bool(x.expected), x.id  # unanswerables have no sources


def test_eval_set_sources_exist():
    from care_rag.sources import load_sources

    ids = {s.id for s in load_sources()}
    for x in load_eval_set():
        assert x.relevant <= ids, (x.id, x.relevant - ids)


def test_cited_numbers_forms():
    from care_rag.answer import cited_numbers

    assert cited_numbers("A [1]. B [2][3]. C [1, 4]. D [3,5]") == [1, 2, 3, 4, 5]
    assert cited_numbers("no citations, see section [a]") == []
