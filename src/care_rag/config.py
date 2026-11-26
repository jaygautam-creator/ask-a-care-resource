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


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    mode: str = "hybrid"  # "dense" | "bm25" | "hybrid"
    rerank: bool = False
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    top_k: int = 5  # chunks given to the LLM
    candidates: int = 30  # per-retriever depth before fusion / reranking
    rrf_k: int = 60  # standard Reciprocal Rank Fusion constant

    def as_dict(self) -> dict:
        return asdict(self)


EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# Configurations compared in the experiment. `default` is what the app serves.
CONFIGS: dict[str, RetrievalConfig] = {
    "dense": RetrievalConfig(name="dense", mode="dense"),
    "bm25": RetrievalConfig(name="bm25", mode="bm25"),
    "hybrid": RetrievalConfig(name="hybrid", mode="hybrid"),
    "hybrid_rerank": RetrievalConfig(name="hybrid_rerank", mode="hybrid", rerank=True),
    "hybrid_small_chunks": RetrievalConfig(
        name="hybrid_small_chunks", mode="hybrid", chunk=ChunkConfig(150, 30, 25)
    ),
}
DEFAULT_CONFIG = "hybrid_rerank"
