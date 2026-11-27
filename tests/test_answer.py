"""answer(): verify-and-repair, script check, Hindi numbers. The LLM and retriever are faked."""

import pytest

import care_rag.answer as A
from care_rag.chunk import Chunk
from care_rag.llm import LLMResponse
from care_rag.retrieve import Hit

PASSAGE = "Free health cover of Rs 5 lakh per year, plus a top-up of up to Rs 5 lakh. Age 70+."


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, system, prompt, json_mode=False):
        self.prompts.append(prompt)
        return LLMResponse(self.replies.pop(0), "fake", "fake-1", cached=False, latency_ms=5.0)


@pytest.fixture
def fake(monkeypatch):
    hit = Hit(Chunk("d#000", "d", "Vay Vandana", "u", "", PASSAGE, "en", "c"), 1.0, dense_sim=0.9)
    monkeypatch.setattr(A, "retrieve_multi", lambda queries, cfg: [hit])

    def install(*replies):
        llm = FakeLLM(replies)
        monkeypatch.setattr(A, "get_llm", lambda role="answer": llm)
        return llm

    return install


def test_clean_answer_is_not_retried(fake):
    llm = fake("Cover is Rs 5 lakh [1].")
    res = A.answer("How much cover?", repair=True)
    assert not res.draft_issues and not res.repaired and len(llm.prompts) == 1


def test_fabricated_number_is_repaired(fake):
    llm = fake("Cover is Rs 7 lakh [1].", "Cover is Rs 5 lakh, plus a 5 lakh top-up [1].")
    res = A.answer("How much cover?", repair=True)
    assert res.repaired and res.unverified_numbers == [] and "7 lakh" in res.draft_issues[0]
    assert "7 lakh" in llm.prompts[1] and res.answer.startswith("Cover is Rs 5 lakh")
    assert "repair" in res.latency_ms


def test_worse_retry_is_discarded(fake):
    fake("Cover is Rs 7 lakh [1].", "Cover is Rs 9 lakh and 8 lakh [1].")
    res = A.answer("How much cover?", repair=True)
    assert not res.repaired and res.unverified_numbers == ["7 lakh"]


def test_retry_that_refuses_or_drops_citations_is_discarded(fake):
    fake("Cover is Rs 7 lakh [1].", A.REFUSAL)
    assert not A.answer("How much cover?", repair=True).repaired
    fake("Cover is Rs 7 lakh [1].", "Cover is Rs 5 lakh.")
    assert not A.answer("How much cover?", repair=True).repaired


def test_repair_can_be_turned_off(fake):
    llm = fake("Cover is Rs 7 lakh [1].")
    res = A.answer("How much cover?", repair=False)
    assert res.draft_issues and not res.repaired and len(llm.prompts) == 1
    assert res.unverified_numbers == ["7 lakh"]


def test_script_mismatch():
    q = "क्या उन्हें आयुष्मान कार्ड से मुफ्त इलाज मिलेगा?"
    assert A.script_mismatch(q, "Haan, unko muft ilaaj milega [1].")
    assert not A.script_mismatch(q, "हाँ, आयुष्मान वय वंदना कार्ड (AB PM-JAY) से मिलेगा [1]।")
    assert not A.script_mismatch("mummy ka card banega?", "Haan, banega [1].")


def test_hindi_numbers_are_checked():
    hits = [Hit(Chunk("d#000", "d", "t", "u", "", PASSAGE, "en", "c"), 1.0)]
    assert A.unverified_numbers("5 लाख रुपये तक [1]।", hits) == []
    assert A.unverified_numbers("७ लाख रुपये तक [1]।", hits) == ["7 lakh"]


def test_retrieve_multi_fuses_rankings(monkeypatch):
    import care_rag.retrieve as R
    from care_rag.config import CONFIGS

    def mk(cid, sim):
        return Hit(Chunk(cid, cid.split("#")[0], "t", "u", "", "x", "en", "c"), 0.0, dense_sim=sim)

    lists = {"q": [mk("a#0", 0.80), mk("b#0", 0.81)], "rw": [mk("b#0", 0.86), mk("c#0", 0.90)]}
    monkeypatch.setattr(R, "retrieve", lambda q, cfg: lists[q])
    hits = R.retrieve_multi(["q", "rw"], CONFIGS["hybrid_rerank_small"])
    assert [h.chunk.chunk_id for h in hits] == ["b#0", "a#0", "c#0"]  # b is in both lists
    assert hits[0].dense_sim == 0.86  # best similarity to any query, for the refusal gate
