# Ask-a-Care-Resource

A retrieval-augmented assistant for families in India who are caring for an elderly parent. It
answers questions about **government schemes, pensions, health cover, helplines and legal rights
for senior citizens**, in English, Hinglish or Hindi. It **cites the passages it used**, and it
says *"I don't know from these sources."* when the corpus does not contain the answer. An honest
evaluation is included, with all of its weak spots.

```
$ python -m care_rag.cli ask "mummy 70 ki ho gayi hai, kya unka free ilaaj wala card banega?"
```

<!-- RESULTS-SUMMARY -->

---

## Run it

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/). It was tested on macOS (M2) and
runs on CPU too. You need one LLM key: a free Gemini key from
<https://aistudio.google.com/apikey> and/or a free Groq key from <https://console.groq.com/keys>.
Alternatively, run Ollama locally.

```bash
make setup     # venv + install; creates .env from .env.example -> put your key(s) in .env
make all       # ingest (clean -> chunk -> embed -> index) + full evaluation + results/RESULTS.md
```

Then:

```bash
make app       # Streamlit UI on http://localhost:8501
make api       # FastAPI on http://localhost:8000/docs   (POST /ask {"question": "..."})
python -m care_rag.cli ask "What is the Elderline number?"     # one question from the shell
make test      # unit tests (no network, no keys needed)
```

Notes:
- **No API key is needed for ingestion or the retrieval evaluation.** `python -m care_rag.cli
  eval --retrieval-only` runs fully offline once the embedding models are downloaded.
- **The raw corpus snapshot is committed** in `data/raw/` with a SHA-256 manifest. Several
  government sites were down or flaky while this was built, and the snapshot makes runs
  reproducible. `python -m care_rag.cli ingest --refresh` re-downloads everything.
- **LLM responses are cached** in `.cache/llm_cache.sqlite` (gitignored). Re-running the
  evaluation costs nothing and gives the same numbers. Delete `.cache/` to force fresh calls.
- **Keys live only in `.env`, which is gitignored.** They are never logged, cached or committed.

## How it works

```
sources.yaml ─▶ fetch ─▶ data/raw (committed snapshot + sha256 manifest)
                           │
                           ▼
              clean: HTML → markdown (trafilatura), PDF → markdown (pymupdf4llm)
                     → sections by heading, PDF word-split repair, boilerplate removal
                           │
                           ▼
              chunk: per section, ~300 words, 50-word sentence-aligned overlap
                     metadata: source id, title, section, url, language, category
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
      multilingual-e5-base      BM25 (rank_bm25)
      → FAISS (exact, cosine)        │
                └──────────┬──────────┘
                           ▼
           Reciprocal Rank Fusion (k = 60) → top 20
                           ▼
           cross-encoder rerank (bge-reranker-v2-m3) → top 5
                           ▼
       refusal gate: best cosine < threshold → "I don't know from these sources."
                           ▼
       LLM (Gemini → Groq → Ollama fallback), numbered passages, strict prompt
                           ▼
       citation check: every [n] must point at a passage that was shown
```

Code map (`src/care_rag/`):

| File | What it does |
|---|---|
| `fetch.py` | Polite fetching with retries and verified TLS; writes the raw snapshot and manifest |
| `clean.py` | HTML/PDF → markdown → sections. Repairs PDF kerning splits ("for m aintenance" → "for maintenance") using corpus word counts |
| `chunk.py` | Section-bounded chunking with sentence-aligned overlap |
| `embed.py`, `index.py` | e5 embeddings (`query:` / `passage:` prefixes), FAISS and BM25 indexes, one per chunk config |
| `retrieve.py` | Dense, BM25, RRF fusion, optional cross-encoder rerank |
| `llm.py` | Gemini / Groq / Ollama behind one interface, fallback, SQLite cache, timeouts |
| `prompts.py` | Every prompt, in one reviewable file |
| `answer.py` | Optional query rewrite → retrieve → gate → generate → citation validation |
| `evaluate.py`, `report.py` | Metrics, LLM judge, gate calibration, `results/RESULTS.md` |
| `cli.py`, `../app.py`, `../api.py` | CLI, Streamlit UI, FastAPI |

## Corpus

There are 29 official documents, about 42,500 words after cleaning. They cover:
- central schemes (PIB releases and backgrounders, PM-JAY guidelines, DGHS, PFRDA, the Income
  Tax Department, SBI for SCSS);
- the full **Maintenance and Welfare of Parents and Senior Citizens Act, 2007**;
- state pensions for **Delhi, Kerala and Maharashtra**;
- one Hindi document, the Hindi edition of PIB's *Elderly in India 2025*.

The full list with URLs is in [`sources.yaml`](sources.yaml) and in the
[appendix](#appendix-sources).

What was left out, and why:
- **myScheme.gov.in**: pages are rendered client-side and the data sits behind an API key
  embedded in their frontend. I did not use that key.
- **India Code, nha.gov.in, socialjustice.gov.in, nsap.nic.in**: these were down or timing out
  during the build. The same facts were taken from PIB copies.
- **ALIMCO, some state portals**: these fail TLS certificate verification. I did not disable
  verification to get them.

## Design choices

**Chunking.** I split each document into its own sections first (headings from HTML, PDF
headings, numbered sections of the Act). Inside a section I pack whole paragraphs and sentences
up to about 300 words, with about 50 words of sentence-aligned overlap. A chunk never crosses a
section boundary. This is because government documents are a list of separate schemes and
clauses: mixing two schemes in one chunk is the main way a RAG system ends up quoting the right
number for the wrong scheme. The overlap keeps a fact that falls across a boundary retrievable.
The embedded text is prefixed with `title | section`, so a short chunk like "Rs 2500 p.m. for
+70 beneficiaries" still carries "Delhi old age assistance" with it.

**Hybrid retrieval and reranking.**
- *Dense retrieval* (multilingual-e5) handles paraphrase and Hindi or Hinglish.
- *BM25* handles exact tokens such as "80TTB", "194P" and "14567".
- *RRF* combines the two rankings without calibrating their scores against each other.
- *A multilingual cross-encoder* re-scores the top 20.

All four choices were measured rather than assumed; see the results.

**Refusal has two layers.**
1. A **similarity gate**: if even the best passage is far from the question, refuse without
   calling the LLM.
2. A **strict prompt**: when on-topic passages don't contain the answer, the LLM must reply with
   the exact refusal sentence, and it must not answer a nearby question instead.

The gate threshold is *calibrated on the eval set*. It is set to the highest value that never
refuses an answerable question, because a wrong refusal costs a family more than an extra LLM
call.

**Citations are checked mechanically.** Every `[n]` must refer to a passage that was shown, and
an answer with no citation at all is flagged.

**A different model family judges.**
- Answers come from Gemini first, with Groq as the fallback.
- The judge is Groq's `gpt-oss-120b` first, so the model grading an answer is, by default, not
  the one that wrote it.
- Temperature is 0 everywhere, and every response is cached.

**Dates matter.** Amounts and schemes change; PMVVY, for example, closed on 31 March 2023. The
prompt asks the model to state the date the source gives, and the eval contains a trap question
on PMVVY.

## Evaluation

[`eval/eval_set.yaml`](eval/eval_set.yaml) contains **43 hand-written questions**:

| Group | Count |
|---|---|
| Formal, single source | 21 |
| Formal, multi-source | 2 |
| Stressed / informal English | 4 |
| Hinglish | 5 |
| Hindi (Devanagari) | 2 |
| **Unanswerable** (the corpus does not contain the answer) | **9** (5 formal, 2 stressed, 1 Hinglish, 1 Hindi) |

The unanswerable questions are chosen to be *tempting*: each one touches something the corpus
does mention, without containing the answer. Examples:
- a railway station in a story, but no railway concession;
- the SCSS interest rate, but no FD rate;
- Tamil Nadu in an Elderline list, but no Tamil Nadu pension;
- dementia, but no care-home prices.

**Expected sources are labelled at document level, as groups.** Each question lists groups of
interchangeable documents; for example, nine PIB documents all give the Elderline number. A
multi-source question has two groups and needs a document from both. Document labels stay valid
across every chunking configuration, whereas chunk IDs do not.

**How each number is computed:**

| Metric | Definition |
|---|---|
| hit@k | An expected document is among the top-k *distinct* documents retrieved |
| recall@5 | Fraction of the question's expected groups covered in the top 5 (matters for multi-source questions) |
| MRR | 1 / rank of the first expected document (0 if none in the top 10) |
| Refusal accuracy | Share of the 9 unanswerable questions that were refused |
| False refusal rate | Share of the 34 answerable questions that were refused |
| Key-fact recall | Share of hand-written key facts (numbers, names) present in the answer after normalising `₹`/`Rs`, commas and case. Objective, but only checks what I wrote down |
| Faithfulness | LLM judge splits the answer into atomic claims; supported claims / all claims, judged *only* against the passages shown |
| Citation correctness | Judge: the passage cited next to a claim actually contains it |
| Citations valid | No `[n]` outside the shown passages, and at least one citation |
| Cited-doc precision | Cited passages that come from an expected document |

<!-- RESULTS -->

## What I'd do next

<!-- NEXT -->

## AI tools used

This project was built with **Claude Code**, Anthropic's coding agent, working as a pair
programmer:
- it scaffolded the code, wrote the cleaning, chunking and evaluation modules, and debugged the
  PDF extraction;
- it drafted the eval set from facts it looked up in the cleaned corpus.

My own work was to choose the scope (India, the Gemini → Groq → Ollama order), and to review,
edit and own the eval questions (especially the Hinglish and Hindi phrasings) and the final
observations. Every number in this README comes from `results/`, produced by the code in this
repository.

## Time spent

<!-- HOURS -->

## Appendix: sources

| id | publisher | lang | document |
|---|---|---|---|
<!-- SOURCES -->
