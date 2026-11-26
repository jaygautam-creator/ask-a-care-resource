"""Streamlit UI: `make app` (or `streamlit run app.py`)."""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st  # noqa: E402

from care_rag.answer import GATE_THRESHOLD, AnswerResult, answer  # noqa: E402
from care_rag.config import CONFIGS, DEFAULT_CONFIG  # noqa: E402
from care_rag.llm import get_llm  # noqa: E402
from care_rag.sources import load_sources  # noqa: E402

st.set_page_config(page_title="Ask-a-Care-Resource", page_icon="🤝", layout="centered")

EXAMPLES = {
    "🏥 Free treatment": "My mother is 74. Is there a government scheme that pays for "
    "her hospital treatment?",
    "🏠 Thrown out": "beta ghar se nikal diya hai papa ko, unke paas paise nahi hai, "
    "kya kar sakte hain?",
    "📞 हेल्पलाइन": "वरिष्ठ नागरिकों के लिए हेल्पलाइन नंबर क्या है?",
    "💰 Delhi pension": "How much old age pension does Delhi pay to a 72-year-old?",
    "🚆 Rail discount": "What concession do senior citizens get on railway tickets?",
}
CONFIG_LABELS = {
    "hybrid_rerank_small": "Hybrid + reranker (recommended)",
    "hybrid_rerank": "Hybrid + large reranker (slower)",
    "hybrid": "Hybrid, no reranker (fastest)",
    "dense": "Dense only",
    "bm25": "Keyword (BM25) only",
    "hybrid_small_chunks": "Hybrid, small chunks",
}

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 820px;}
.src-card {border: 1px solid rgba(128,128,128,.25); border-radius: 10px;
           padding: .6rem .8rem; margin-bottom: .5rem;}
.src-card .n {font-weight: 700; margin-right: .35rem;}
.src-card .sec {opacity: .7; font-size: .85rem;}
.muted {opacity: .65; font-size: .85rem;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading search models (first time only)…")
def warm_up(cfg_name: str) -> None:
    """Load the index, embedder and reranker once per process, not on the first question."""
    from care_rag.index import load_index
    from care_rag.retrieve import get_reranker, retrieve

    cfg = CONFIGS[cfg_name]
    load_index(cfg.chunk)
    if cfg.reranker:
        get_reranker(cfg.reranker)
    retrieve("senior citizen pension", cfg)


# ------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Settings")
    names = [n for n in CONFIG_LABELS if n in CONFIGS]
    cfg_name = st.selectbox("Search method", names, index=names.index(DEFAULT_CONFIG),
                            format_func=CONFIG_LABELS.get)
    rewrite = st.toggle("Rewrite question before search", value=False,
                        help="An LLM turns Hinglish/Hindi or vague questions into a clear "
                             "English search query. Costs one extra LLM call.")
    gate = st.slider("Refusal gate (cosine similarity)", 0.70, 0.90, GATE_THRESHOLD, 0.005,
                     help="Refuse without calling the LLM if no passage is at least this "
                          "similar to the question.")
    show_passages = st.toggle("Show retrieved passages", value=False)
    if st.button("Clear conversation", width="stretch"):
        st.session_state.history = []
        st.rerun()

    st.divider()
    sources = load_sources()
    st.subheader("What it knows")
    st.caption(f"{len(sources)} official documents: PIB, Ministry of Health, PFRDA, Income Tax "
               "Department, the MWPSC Act 2007, and state pages for Delhi, Kerala and "
               "Maharashtra.")
    with st.expander("All sources"):
        for s in sources:
            st.markdown(f"- [{s.title}]({s.url})")
    llm = get_llm("answer")
    st.caption(f"Answer model: `{llm.describe()}`")

# ---------------------------------------------------------------------------------- main
st.title("Ask-a-Care-Resource")
st.markdown("Ask about **pensions, health cover, helplines and legal rights** for an elderly "
            "parent in India, in English, Hinglish or Hindi. Every answer cites the official "
            "source it came from. If the sources don't cover it, it says so.")

if not llm.providers:
    st.error("No LLM is configured. Add `GEMINI_API_KEY` or `GROQ_API_KEY` to `.env` "
             "(see `.env.example`) or start Ollama, then reload.")
    st.stop()

warm_up(cfg_name)

st.session_state.setdefault("history", [])
picked = None
cols = st.columns(len(EXAMPLES))
for col, (label, q) in zip(cols, EXAMPLES.items(), strict=True):
    if col.button(label, help=q, width="stretch"):
        picked = q


def render(res: AnswerResult) -> None:
    if res.refused:
        st.warning(res.answer)
        why = ("no indexed passage is close to this question" if res.refusal_reason == "gate"
               else "the retrieved passages do not contain the answer")
        st.markdown(f"<span class='muted'>Refused because {why}. For help, Elderline "
                    "<b>14567</b> is the national toll-free helpline for senior citizens.</span>",
                    unsafe_allow_html=True)
    else:
        st.markdown(res.answer)
        if res.invalid_citations:
            st.error(f"Citation check failed: {res.invalid_citations} do not match any passage.")
        elif not res.citations:
            st.error("Citation check failed: this answer cites no source.")
        if res.unverified_numbers:
            st.warning("Could not find these numbers in the cited sources: "
                       f"{', '.join(res.unverified_numbers)}. Please check them in the "
                       "source before relying on them.", icon="⚠️")

    if res.citations:
        st.markdown("**Sources**")
        for c in res.citations:
            st.markdown(
                f"<div class='src-card'><span class='n'>[{c.n}]</span>"
                f"<a href='{c.url}' target='_blank'>{c.title}</a><br>"
                f"<span class='sec'>{c.section or 'Introduction'}</span></div>",
                unsafe_allow_html=True,
            )

    if show_passages:
        with st.expander("Retrieved passages"):
            if res.search_query != res.question:
                st.caption(f"Search query after rewrite: {res.search_query}")
            for n, h in enumerate(res.hits, 1):
                st.markdown(f"**[{n}] {h.chunk.title}** — {h.chunk.section or 'intro'}")
                st.caption(f"score {h.score:.3f} · cosine {h.dense_sim:.3f} · dense rank "
                           f"{h.ranks.get('dense')} · BM25 rank {h.ranks.get('bm25')}")
                st.text(h.chunk.text[:1000])

    lat = " · ".join(f"{k} {v / 1000:.1f}s" for k, v in res.latency_ms.items()
                     if not (res.cached and k in ("generate", "total")))
    model = f"{res.provider}:{res.model}" if res.provider else "no LLM call"
    if res.cached:
        model += " (cached answer)"
    st.markdown(f"<span class='muted'>{model} · {lat}</span>", unsafe_allow_html=True)


for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn.question)
    with st.chat_message("assistant", avatar="🤝"):
        render(turn)

question = st.chat_input("e.g. papa 81 saal ke hain, BPL card hai, kitni pension milegi?")
# A shareable link such as /?q=helpline+number opens the app with that question answered.
linked = st.query_params.get("q") if not st.session_state.get("linked_done") else None
st.session_state["linked_done"] = True
question = (question or picked or linked or "").strip()
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🤝"):
        with st.spinner("Searching the sources…"):
            try:
                res = answer(question, CONFIGS[cfg_name], gate=gate, rewrite=rewrite)
            except Exception as exc:  # show provider errors instead of a stack trace
                st.error(f"Could not answer right now: {exc}")
                st.stop()
        render(res)
    st.session_state.history.append(res)

st.caption("Information comes from official government pages as indexed on the date shown in "
           "the source. Amounts and rules change, so confirm with the office before applying.")
