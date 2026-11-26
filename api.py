"""FastAPI service: `uvicorn api:app` (or `make api`), then POST /ask or open /docs."""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from care_rag.answer import answer  # noqa: E402
from care_rag.config import CONFIGS, DEFAULT_CONFIG  # noqa: E402
from care_rag.index import load_index  # noqa: E402

app = FastAPI(title="Ask-a-Care-Resource", version="1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    config: str = DEFAULT_CONFIG
    rewrite: bool = False


class Source(BaseModel):
    n: int
    title: str
    section: str
    url: str
    chunk_id: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None
    sources: list[Source]
    invalid_citations: list[int]
    provider: str | None
    latency_ms: dict


@app.get("/health")
def health() -> dict:
    idx = load_index(CONFIGS[DEFAULT_CONFIG].chunk)
    return {"status": "ok", "chunks": len(idx.chunks), "default_config": DEFAULT_CONFIG}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if req.config not in CONFIGS:
        raise HTTPException(400, f"unknown config; choose from {sorted(CONFIGS)}")
    try:
        res = answer(req.question, CONFIGS[req.config], rewrite=req.rewrite)
    except RuntimeError as exc:  # LLMError: no provider configured / all failed
        raise HTTPException(503, str(exc)) from exc
    return AskResponse(
        answer=res.answer,
        refused=res.refused,
        refusal_reason=res.refusal_reason,
        sources=[Source(n=c.n, title=c.title, section=c.section, url=c.url,
                        chunk_id=c.chunk_id) for c in res.citations],
        invalid_citations=res.invalid_citations,
        provider=f"{res.provider}:{res.model}" if res.provider else None,
        latency_ms={k: round(v) for k, v in res.latency_ms.items()},
    )
