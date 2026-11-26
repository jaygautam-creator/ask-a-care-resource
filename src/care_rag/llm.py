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
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

import httpx

from care_rag.config import CACHE_DIR

# Per-request timeout: a hung request then falls through to the next provider.
TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "60"))


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

    _client = None

    @property
    def client(self):
        # Keep one client alive: a temporary genai.Client is garbage-collected (and its HTTP
        # session closed) while the request is still being sent.
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=os.environ["GEMINI_API_KEY"],
                http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000),  # milliseconds
            )
        return self._client

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

        client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=TIMEOUT_S, max_retries=0)
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
        # flash-lite: the free tier's daily quota for gemini-3.5-flash ran out after ~20 calls,
        # too few for an evaluation run (43 questions x 3 configurations).
        "gemini": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        "groq": os.getenv(
            "GROQ_JUDGE_MODEL" if role == "judge" else "GROQ_MODEL", "openai/gpt-oss-120b"
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

    def get(self, key: str) -> tuple[str, float] | None:
        """(text, latency of the original uncached call in ms)."""
        with self._lock:
            row = self._db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        try:
            v = json.loads(row[0])
            return v["text"], float(v["latency_ms"])
        except (ValueError, TypeError, KeyError):  # entries written before latency was stored
            return row[0], 0.0

    def put(self, key: str, text: str, latency_ms: float) -> None:
        value = json.dumps({"text": text, "latency_ms": round(latency_ms, 1)})
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, value))
            self._db.commit()


# ---------------------------------------------------------------------------------- the chain


RETRY_IN_RE = re.compile(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", re.I)
MAX_WAIT_S = 120  # a provider asking us to wait longer than this is skipped


def redact(msg: str) -> str:
    """Error text can end up in results files: strip account identifiers and key-like strings."""
    msg = re.sub(r"org_[A-Za-z0-9]+", "org_<redacted>", msg)
    return re.sub(r"(AIza|gsk_)[A-Za-z0-9_-]{10,}", r"\1<redacted>", msg)


def retry_wait(msg: str, attempt: int) -> float | None:
    """Seconds to wait before retrying this error, or None if retrying is pointless."""
    low = msg.lower()
    if "exceeded your current quota" in low:  # Gemini daily quota: gone until tomorrow
        return None
    if m := RETRY_IN_RE.search(msg):  # Groq says exactly how long: "try again in 1m2.5s"
        wait = int(m.group(1) or 0) * 60 + float(m.group(2)) + 1
        return wait if wait <= MAX_WAIT_S else None
    if any(t in low for t in ("429", "rate", "quota", "503", "unavailable", "overloaded")):
        return 15.0 * (attempt + 1)
    return None


class LLM:
    """Try each configured provider in order; cache successful responses."""

    def __init__(self, providers: list[Provider], use_cache: bool = True, attempts: int = 3):
        self.providers = providers
        self.cache = ResponseCache() if use_cache else None
        self.attempts = attempts

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
                # report the original call's latency, so cached reruns keep honest timings
                return LLMResponse(hit[0], p.name, p.model, cached=True, latency_ms=hit[1])
            for attempt in range(self.attempts):
                t0 = time.perf_counter()
                try:
                    text = p.generate(system, prompt, json_mode)
                except Exception as exc:  # provider SDKs raise many types; we fall through
                    msg = str(exc)
                    wait = retry_wait(msg, attempt)
                    if wait is not None and attempt < self.attempts - 1:
                        time.sleep(wait)  # free tiers: per-minute limits, overload
                        continue
                    errors.append(f"{p.name}: {type(exc).__name__}: {redact(msg)[:200]}")
                    break
                latency = (time.perf_counter() - t0) * 1000
                if self.cache:
                    self.cache.put(key, text, latency)
                return LLMResponse(text, p.name, p.model, cached=False, latency_ms=latency)
        raise LLMError("all providers failed: " + " | ".join(errors))


ORDER = {"answer": ["gemini", "groq", "ollama"], "judge": ["groq", "gemini", "ollama"]}


@lru_cache(maxsize=4)
def get_llm(role: str = "answer") -> LLM:
    order = os.getenv(f"{role.upper()}_PROVIDERS", ",".join(ORDER[role])).split(",")
    providers = [_provider(n.strip(), role) for n in order if n.strip()]
    # The judge waits longer for its first choice (Groq) before falling back to Gemini, because
    # falling back means Gemini grades Gemini's own answers (flagged as self_judged).
    use_cache = os.getenv("LLM_CACHE", "1") != "0"  # LLM_CACHE=0: always call the API
    return LLM([p for p in providers if p.available()], use_cache=use_cache,
               attempts=6 if role == "judge" else 3)


def parse_json(text: str) -> dict:
    """Parse a JSON object from an LLM reply, tolerating ```json fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in: {text[:200]}")
    return json.loads(text[start : end + 1])
