"""Build and load the per-chunk-config index: chunks + FAISS (dense) + BM25 (sparse).

One index directory per chunk configuration (data/index/<chunk-config-name>/), so the
chunk-size experiment never mixes vectors from different chunkings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from care_rag.chunk import Chunk, chunk_documents, load_chunks, save_chunks
from care_rag.clean import load_documents
from care_rag.config import EMBED_MODEL, INDEX_DIR, ChunkConfig
from care_rag.embed import embed_passages

# Small English stop list: BM25 should match on "pension", "70", "tribunal", not on "the".
STOPWORDS = set(
    "a an the of for to in on at by and or is are was were be been it its this that with as from "
    "what which who how can do does i my me we our you your he she his her they their there "
    "will shall may any all not no".split()
)
# Python's \w does NOT match Devanagari vowel signs (matras), which would shred "पेंशन" into
# letters, so the whole Devanagari block is added explicitly.
TOKEN_RE = re.compile(r"[\wऀ-ॿ]+")


def bm25_tokenize(text: str) -> list[str]:
    tokens = []
    for tok in TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]  # crude plural folding: "pensions" == "pension"
        tokens.append(tok)
    return tokens


@dataclass
class Index:
    chunks: list[Chunk]
    faiss_index: faiss.Index
    bm25: BM25Okapi
    chunk_cfg: ChunkConfig


def index_dir(cfg: ChunkConfig):
    return INDEX_DIR / cfg.name


def build_index(cfg: ChunkConfig) -> Index:
    docs = load_documents()
    chunks = chunk_documents(docs, cfg)
    out = index_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    save_chunks(chunks, out / "chunks.jsonl")
    vecs = embed_passages([c.embed_text for c in chunks])
    np.save(out / "embeddings.npy", vecs)
    fi = faiss.IndexFlatIP(vecs.shape[1])  # exact search; ~1-2k vectors needs nothing fancier
    fi.add(vecs)
    faiss.write_index(fi, str(out / "faiss.index"))
    (out / "meta.json").write_text(
        json.dumps({"embed_model": EMBED_MODEL, "chunk": cfg.name, "n_chunks": len(chunks)})
    )
    load_index.cache_clear()
    return load_index(cfg)


@lru_cache(maxsize=4)
def load_index(cfg: ChunkConfig) -> Index:
    d = index_dir(cfg)
    if not (d / "faiss.index").exists():
        raise FileNotFoundError(f"no index at {d}; run `python -m care_rag.cli ingest` first")
    meta = json.loads((d / "meta.json").read_text())
    if meta["embed_model"] != EMBED_MODEL:
        raise RuntimeError(f"index built with {meta['embed_model']}, config says {EMBED_MODEL}")
    chunks = load_chunks(d / "chunks.jsonl")
    bm25 = BM25Okapi([bm25_tokenize(c.embed_text) for c in chunks])
    return Index(chunks, faiss.read_index(str(d / "faiss.index")), bm25, cfg)
