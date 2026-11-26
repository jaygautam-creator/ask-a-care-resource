"""Retrieval: dense (FAISS), sparse (BM25), hybrid (Reciprocal Rank Fusion), optional rerank.

RRF: score(chunk) = sum over retrievers of 1 / (k + rank). It needs no score calibration
between cosine similarity and BM25 (which live on different scales), and k=60 is the value
from the original RRF paper (Cormack et al., 2009).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from care_rag.chunk import Chunk
from care_rag.config import RetrievalConfig
from care_rag.embed import embed_query
from care_rag.index import Index, bm25_tokenize, load_index


@dataclass
class Hit:
    chunk: Chunk
    score: float  # the score of the final stage (rerank logit, RRF score, cosine or BM25)
    dense_sim: float | None = None  # cosine similarity to the query, kept for the refusal gate
    ranks: dict = field(default_factory=dict)  # per-retriever rank, for debugging / the UI


def dense_search(index: Index, query: str, k: int) -> list[tuple[int, float]]:
    sims, ids = index.faiss_index.search(embed_query(query), k)
    return [(int(i), float(s)) for i, s in zip(ids[0], sims[0], strict=True) if i != -1]


def bm25_search(index: Index, query: str, k: int) -> list[tuple[int, float]]:
    scores = index.bm25.get_scores(bm25_tokenize(query))
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


def rrf(rank_lists: list[list[int]], k: int) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


@lru_cache(maxsize=2)
def get_reranker(name: str):
    from sentence_transformers import CrossEncoder

    from care_rag.embed import _device

    return CrossEncoder(name, device=_device(), max_length=512)


def retrieve(query: str, cfg: RetrievalConfig, index: Index | None = None) -> list[Hit]:
    index = index or load_index(cfg.chunk)
    n = cfg.candidates
    dense = dense_search(index, query, n) if cfg.mode in ("dense", "hybrid") else []
    sparse = bm25_search(index, query, n) if cfg.mode in ("bm25", "hybrid") else []
    dense_sim = dict(dense)
    dense_rank = {i: r for r, (i, _) in enumerate(dense, 1)}
    sparse_rank = {i: r for r, (i, _) in enumerate(sparse, 1)}

    if cfg.mode == "dense":
        ordered = dense
    elif cfg.mode == "bm25":
        ordered = sparse
    else:
        fused = rrf([[i for i, _ in dense], [i for i, _ in sparse]], cfg.rrf_k)
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:n]

    if cfg.reranker and ordered:
        cand = [i for i, _ in ordered[: cfg.rerank_depth]]
        pairs = [(query, index.chunks[i].embed_text) for i in cand]
        scores = get_reranker(cfg.reranker).predict(pairs, batch_size=16)
        ordered = sorted(zip(cand, map(float, scores), strict=True), key=lambda kv: -kv[1])

    # Dense similarity is needed for the refusal gate even in BM25 mode: compute if missing.
    hits = []
    qv = None
    for i, score in ordered[: cfg.top_k]:
        if i not in dense_sim:
            qv = embed_query(query) if qv is None else qv
            dense_sim[i] = float(index.faiss_index.reconstruct(i) @ qv[0])
        hits.append(
            Hit(
                chunk=index.chunks[i],
                score=score,
                dense_sim=dense_sim[i],
                ranks={"dense": dense_rank.get(i), "bm25": sparse_rank.get(i)},
            )
        )
    return hits
