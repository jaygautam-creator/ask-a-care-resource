"""Streamlit UI: `streamlit run app.py` (or `make app`)."""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st  # noqa: E402

from care_rag.answer import GATE_THRESHOLD, answer  # noqa: E402
from care_rag.config import CONFIGS, DEFAULT_CONFIG  # noqa: E402
from care_rag.llm import get_llm  # noqa: E402

st.set_page_config(page_title="Ask-a-Care-Resource", page_icon="🧓", layout="wide")

EXAMPLES = [
    "My mother is 74. Is there a government scheme that pays for her hospital treatment?",
    "beta ghar se nikal diya hai papa ko, kya kar sakte hain?",
    "वरिष्ठ नागरिकों के लिए हेल्पलाइन नंबर क्या है?",
    "How much old age pension does Delhi pay to a 72-year-old?",
    "What concession do senior citizens get on railway tickets?",
]

with st.sidebar:
    st.header("Settings")
    names = list(CONFIGS)
    cfg_name = st.selectbox("Retrieval", names, index=names.index(DEFAULT_CONFIG))
    rewrite = st.toggle("LLM query rewrite (helps Hinglish)", value=False)
    gate = st.slider("Refusal gate (cosine)", 0.70, 0.90, GATE_THRESHOLD, 0.005,
                     help="Refuse without calling the LLM if no passage is this similar.")
    st.caption(f"Answer model chain: {get_llm('answer').describe()}")
    st.caption("Answers come only from the indexed government sources. "
               "Amounts change: always confirm with the official office or helpline.")

st.title("Ask-a-Care-Resource")
st.write("Questions about pensions, health cover, helplines and legal rights for elderly "
         "parents in India. Every answer cites the official source it came from.")

cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES, strict=True):
    if col.button(ex[:38] + "…", help=ex, use_container_width=True):
        st.session_state["q"] = ex

question = st.text_area("Your question (English, Hinglish or Hindi)",
                        key="q", height=80)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching the sources…"):
        try:
            res = answer(question.strip(), CONFIGS[cfg_name], gate=gate, rewrite=rewrite)
        except Exception as exc:  # show provider/config errors instead of a stack trace
            st.error(f"Could not answer: {exc}")
            st.stop()

    if res.refused:
        st.warning(res.answer)
        why = ("no indexed passage is close to this question" if res.refusal_reason == "gate"
               else "the retrieved passages do not contain the answer")
        st.caption(f"Refused because {why}. For anything urgent, Elderline 14567 is the national "
                   "helpline for senior citizens (from the indexed PIB sources).")
    else:
        st.markdown(res.answer)
        if res.invalid_citations:
            st.error(f"Citation check: {res.invalid_citations} do not match any passage.")

    if res.citations:
        st.subheader("Sources cited")
        for c in res.citations:
            st.markdown(f"**[{c.n}]** [{c.title}]({c.url}) — {c.section or 'introduction'}")

    with st.expander("Retrieved passages and scores"):
        if res.search_query != question.strip():
            st.caption(f"Search query after rewrite: {res.search_query}")
        for n, h in enumerate(res.hits, 1):
            st.markdown(f"**[{n}] {h.chunk.title}** — {h.chunk.section or 'intro'}  \n"
                        f"`score {h.score:.3f} · cosine {h.dense_sim:.3f} · "
                        f"dense rank {h.ranks.get('dense')} · bm25 rank {h.ranks.get('bm25')}`")
            st.text(h.chunk.text[:1200])

    lat = " · ".join(f"{k} {v / 1000:.2f}s" for k, v in res.latency_ms.items())
    st.caption(f"{res.provider or 'no LLM call'} {res.model or ''} · {lat}")
