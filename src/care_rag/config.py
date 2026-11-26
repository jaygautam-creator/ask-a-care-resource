"""Paths and tunable settings in one place.

Everything that an experiment might vary (chunk size, retrieval mode, reranker, models) lives in
`RetrievalConfig` / `Settings`, so a run is fully described by a small, printable object.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")  # secrets stay in .env (gitignored); never hard-coded

SOURCES_FILE = ROOT / "sources.yaml"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # committed snapshot of fetched bytes
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned documents + chunks (rebuilt)
INDEX_DIR = DATA_DIR / "index"  # FAISS / BM25 indexes (rebuilt)
CACHE_DIR = ROOT / ".cache"  # LLM response cache (rebuilt)
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = ROOT / "results"

REFUSAL = "I don't know from these sources."


@dataclass(frozen=True)
class ChunkConfig:
    """Chunk budget in words (~1.3 tokens per English word for the e5 tokenizer)."""

    target_words: int = 300
    overlap_words: int = 50
    min_words: int = 40  # trailing fragments smaller than this are merged into the previous chunk

    @property
    def name(self) -> str:
        return f"w{self.target_words}o{self.overlap_words}"


EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
RERANK_LARGE = "BAAI/bge-reranker-v2-m3"  # 568M params, multilingual; ~3.4 s / 30 pairs on M2
RERANK_SMALL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # 118M, multilingual; ~0.4 s


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    mode: str = "hybrid"  # "dense" | "bm25" | "hybrid"
    reranker: str | None = None  # cross-encoder model name, or None for no reranking
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    top_k: int = 5  # chunks given to the LLM
    candidates: int = 30  # per-retriever depth before fusion
    rerank_depth: int = 20  # how many fused candidates the cross-encoder re-scores
    rrf_k: int = 60  # standard Reciprocal Rank Fusion constant

    def as_dict(self) -> dict:
        return asdict(self)


# Configurations compared in the experiment. DEFAULT_CONFIG is what the app serves.
CONFIGS: dict[str, RetrievalConfig] = {
    "dense": RetrievalConfig(name="dense", mode="dense"),
    "bm25": RetrievalConfig(name="bm25", mode="bm25"),
    "hybrid": RetrievalConfig(name="hybrid", mode="hybrid"),
    "hybrid_rerank": RetrievalConfig(name="hybrid_rerank", mode="hybrid", reranker=RERANK_LARGE),
    "hybrid_rerank_small": RetrievalConfig(
        name="hybrid_rerank_small", mode="hybrid", reranker=RERANK_SMALL
    ),
    "hybrid_small_chunks": RetrievalConfig(
        name="hybrid_small_chunks", mode="hybrid", chunk=ChunkConfig(150, 30, 25)
    ),
}
# Chosen from the retrieval experiment: the small reranker matched or beat bge-reranker-v2-m3
# on every metric at ~10x lower latency (see results/RESULTS.md).
DEFAULT_CONFIG = os.getenv("DEFAULT_CONFIG", "hybrid_rerank_small")
