"""Sentence-transformer embeddings (multilingual-e5).

e5 models are trained with "query: " / "passage: " prefixes; omitting them measurably hurts
retrieval, so the prefixes are applied here and nowhere else. Vectors are L2-normalised, so the
FAISS inner product equals cosine similarity.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from care_rag.config import EMBED_MODEL


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def get_model(name: str = EMBED_MODEL) -> SentenceTransformer:
    return SentenceTransformer(name, device=_device())


def _prefix(name: str, kind: str) -> str:
    return f"{kind}: " if "e5" in name.lower() else ""


def embed_passages(texts: list[str], name: str = EMBED_MODEL) -> np.ndarray:
    model = get_model(name)
    vecs = model.encode(
        [_prefix(name, "passage") + t for t in texts],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 64,
    )
    return np.asarray(vecs, dtype="float32")


def embed_query(text: str, name: str = EMBED_MODEL) -> np.ndarray:
    model = get_model(name)
    vec = model.encode([_prefix(name, "query") + text], normalize_embeddings=True)
    return np.asarray(vec, dtype="float32")
