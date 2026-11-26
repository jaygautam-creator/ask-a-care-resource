from care_rag.chunk import chunk_documents, chunk_section, split_units
from care_rag.clean import Document, Section
from care_rag.config import ChunkConfig
from care_rag.index import bm25_tokenize
from care_rag.retrieve import rrf


def _sentences(n: int, words_each: int = 10) -> str:
    return " ".join(f"Sentence {i} " + "word " * (words_each - 3) + "end." for i in range(n))


def test_chunks_respect_target_and_overlap():
    cfg = ChunkConfig(target_words=50, overlap_words=10, min_words=5)
    chunks = chunk_section(split_units(_sentences(20), cfg.target_words), cfg)
    assert len(chunks) > 1
    assert all(len(c.split()) <= cfg.target_words for c in chunks)
    # consecutive chunks share their boundary sentence (sentence-aligned overlap)
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.splitlines()[-1] == b.splitlines()[0]


def test_short_tail_is_merged_not_orphaned():
    cfg = ChunkConfig(target_words=30, overlap_words=0, min_words=8)
    units = ["one " * 25, "two " * 3]  # second unit is a 3-word tail
    chunks = chunk_section([u.strip() for u in units], cfg)
    assert len(chunks) == 1 and chunks[0].endswith("two two two")


def test_abbreviations_do_not_split_sentences():
    units = split_units("Pension is Rs. 2,500 per month. Apply at the office. " * 30, 20)
    assert not any(u.startswith("2,500") for u in units)


def test_chunks_never_cross_sections_and_carry_heading():
    doc = Document("d", "Title", "u", "c", "en", "p",
                   [Section("A", _sentences(3)), Section("B", _sentences(3))])
    chunks = chunk_documents([doc], ChunkConfig(300, 50, 5))
    assert [c.section for c in chunks] == ["A", "B"]
    assert chunks[1].embed_text.startswith("Title | B\n")


def test_rrf_rewards_agreement_between_retrievers():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    assert max(fused, key=fused.get) == 1  # ranked 1st and 2nd beats 3rd and 1st
    assert fused[3] > fused[2] and 4 in fused


def test_bm25_tokenize_keeps_devanagari_and_folds_plurals():
    assert bm25_tokenize("Pensions for the elderly") == ["pension", "elderly"]
    assert "पेंशन" in bm25_tokenize("वृद्धावस्था पेंशन योजना")
