from collections import Counter

from care_rag.clean import (
    Section,
    _merge_tiny_sections,
    fix_heading_numbers,
    fix_spacing,
    markdown_to_sections,
    repair_split_words,
)

# Corpus counts (merged words are real) and clean reference counts (fragments are not words).
VOCAB = Counter(
    {"maintenance": 50, "india": 40, "for": 900, "form": 30, "may": 300, "who": 60,
     "commencement": 5, "department": 9, "in": 800, "a": 700, "ay": 40, "ent": 30}
)
REF = Counter(
    {"for": 500, "form": 20, "may": 200, "who": 40, "in": 600, "a": 500, "of": 900, "the": 999,
     "apply": 30, "whole": 10, "live": 8, "home": 12}
)


def test_repairs_kerning_splits():
    assert repair_split_words("Application for m aintenance", VOCAB, REF) == (
        "Application for maintenance"
    )
    assert repair_split_words("whole of I ndia", VOCAB, REF) == "whole of India"
    assert repair_split_words("the com m encem ent of", VOCAB, REF) == "the commencement of"
    assert repair_split_words("(Legislative Departm ent)", VOCAB, REF) == (
        "(Legislative Department)"
    )


def test_does_not_merge_real_phrases():
    # "for m" must not become "form": the right merge is "m aintenance".
    assert repair_split_words("for m aintenance", VOCAB, REF) == "for maintenance"
    assert repair_split_words("live in a home", VOCAB, REF) == "live in a home"


def test_does_not_strand_orphan_fragment():
    assert repair_split_words("who m ay apply", VOCAB, REF) == "who may apply"


def test_fix_heading_numbers_and_spacing():
    assert fix_heading_numbers("1 0 . Alteration in allowance") == "10. Alteration in allowance"
    assert fix_heading_numbers("Act, 2 0 0 7") == "Act, 2007"
    assert fix_spacing("a full -scale survey , in ( India )") == "a full-scale survey, in (India)"


def test_markdown_to_sections_splits_on_headings():
    md = "# Title\nintro text\n\n## Eligibility\n- age 60+\n\n**Benefits**\nRs 1000 per month"
    secs = markdown_to_sections(md)
    assert [s.heading for s in secs] == ["Title", "Eligibility", "Benefits"]
    assert secs[1].text == "- age 60+"


def test_fallback_headings_promote_title_case_lines():
    md = "Challenges Faced by Elderly\n\nThe elderly in India often find themselves vulnerable."
    secs = markdown_to_sections(md, fallback_headings=True)
    assert secs[0].heading == "Challenges Faced by Elderly"


def test_tiny_sections_are_folded_into_next():
    secs = [Section("A", "two words"), Section("B", " ".join(["w"] * 30))]
    out = _merge_tiny_sections(secs, min_words=25)
    assert len(out) == 1 and out[0].heading == "B" and out[0].text.startswith("A\ntwo words")
