"""LLM providers behind one interface, with fallback and a response cache.

* Answer chain (user's preference): Gemini -> Groq -> Ollama. A provider is used only if it is
  configured (API key present / Ollama reachable); if a call fails (rate limit, outage), the
  next provider is tried and the answer records which provider actually produced it.
* Judge chain: Groq -> Gemini -> Ollama, i.e. a *different model family* from the answerer by
  default, so the judge is not grading its own outputs.
* Every response is cached in SQLite keyed by (provider, model, system, prompt, json flag).
  Evaluation reruns are free and deterministic, and free-tier rate limits do not break a run.
  Temperature is 0 everywhere.
* Keys come only from the environment (.env, gitignored). They are never logged or cached.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

import httpx

from care_rag.config import CACHE_DIR


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    cached: bool
    latency_ms: float


# ------------------------------------------------------------------------------------ providers


class Provider:
    name = "base"

    def __init__(self, model: str):
        self.model = model

    def available(self) -> bool:
        raise NotImplementedError

    def generate(self, system: str, prompt: str, json_mode: bool) -> str:
        raise NotImplementedError


class GeminiProvider(Provider):
    name = "gemini"

    def available(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY"))

    @property
    def client(self):
        from google import genai

        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate(self, system: str, prompt: str, json_mode: bool) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
        if not resp.text:
            reason = resp.candidates[0].finish_reason if resp.candidates else "unknown"
            raise LLMError(f"gemini returned no text (finish reason: {reason})")
        return resp.text


class GroqProvider(Provider):
    name = "groq"

    def available(self) -> bool:
        return bool(os.getenv("GROQ_API_KEY"))

    def generate(self, system: str, prompt: str, json_mode: bool) -> str:
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            **kwargs,
        )
        return resp.choices[0].message.content or ""


class OllamaProvider(Provider):
    name = "ollama"

    @property
    def host(self) -> str:
        return os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def available(self) -> bool:
        try:
            tags = httpx.get(f"{self.host}/api/tags", timeout=2).json()
        except (httpx.HTTPError, ValueError):
            return False
        return any(m["name"].startswith(self.model) for m in tags.get("models", []))

    def generate(self, system: str, prompt: str, json_mode: bool) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if json_mode:
            body["format"] = "json"
        resp = httpx.post(f"{self.host}/api/chat", json=body, timeout=300)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def _provider(name: str, role: str) -> Provider:
    defaults = {
        "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "groq": os.getenv(
            "GROQ_JUDGE_MODEL" if role == "judge" else "GROQ_MODEL", "llama-3.3-70b-versatile"
        ),
        "ollama": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    }
    cls = {"gemini": GeminiProvider, "groq": GroqProvider, "ollama": OllamaProvider}[name]
    return cls(defaults[name])


# -------------------------------------------------------------------------------------- cache


class ResponseCache:
    def __init__(self, path=CACHE_DIR / "llm_cache.sqlite"):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self._lock = threading.Lock()

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, value))
            self._db.commit()


# ---------------------------------------------------------------------------------- the chain


class LLM:
    """Try each configured provider in order; cache successful responses."""

    def __init__(self, providers: list[Provider], use_cache: bool = True):
        self.providers = providers
        self.cache = ResponseCache() if use_cache else None

    def describe(self) -> str:
        return " -> ".join(f"{p.name}:{p.model}" for p in self.providers) or "(none)"

    def generate(self, system: str, prompt: str, json_mode: bool = False) -> LLMResponse:
        if not self.providers:
            raise LLMError(
                "No LLM configured. Put GEMINI_API_KEY or GROQ_API_KEY in .env, or run Ollama."
            )
        errors = []
        for p in self.providers:
            key = ResponseCache.key(p.name, p.model, system, prompt, str(json_mode))
            if self.cache and (hit := self.cache.get(key)) is not None:
                return LLMResponse(hit, p.name, p.model, cached=True, latency_ms=0.0)
            for attempt in range(3):
                t0 = time.perf_counter()
                try:
                    text = p.generate(system, prompt, json_mode)
                except Exception as exc:  # provider SDKs raise many types; we fall through
                    msg = str(exc)
                    rate_limited = "429" in msg or "rate" in msg.lower() or "quota" in msg.lower()
                    if rate_limited and attempt < 2:
                        time.sleep(15 * (attempt + 1))  # free tiers: per-minute limits
                        continue
                    errors.append(f"{p.name}: {type(exc).__name__}: {msg[:200]}")
                    break
                latency = (time.perf_counter() - t0) * 1000
                if self.cache:
                    self.cache.put(key, text)
                return LLMResponse(text, p.name, p.model, cached=False, latency_ms=latency)
        raise LLMError("all providers failed: " + " | ".join(errors))


ORDER = {"answer": ["gemini", "groq", "ollama"], "judge": ["groq", "gemini", "ollama"]}


@lru_cache(maxsize=4)
def get_llm(role: str = "answer") -> LLM:
    order = os.getenv(f"{role.upper()}_PROVIDERS", ",".join(ORDER[role])).split(",")
    providers = [_provider(n.strip(), role) for n in order if n.strip()]
    return LLM([p for p in providers if p.available()])


def parse_json(text: str) -> dict:
    """Parse a JSON object from an LLM reply, tolerating ```json fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in: {text[:200]}")
    return json.loads(text[start : end + 1])
