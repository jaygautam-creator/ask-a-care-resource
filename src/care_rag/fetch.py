"""Fetch each source once and snapshot the raw bytes to data/raw/.

The snapshot (plus manifest.json with URL, SHA-256, HTTP status and fetch time) is committed, so
the rest of the pipeline is reproducible even when a government site is down. `--refresh`
re-downloads; otherwise existing files are reused. TLS certificates are always verified.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime

import httpx

from care_rag.config import RAW_DIR
from care_rag.sources import Source, load_sources

USER_AGENT = "Mozilla/5.0 (compatible; care-rag-research/0.1; educational RAG project)"
MANIFEST = RAW_DIR / "manifest.json"


def _ext(content: bytes, content_type: str) -> str:
    return "pdf" if content[:4] == b"%PDF" or "pdf" in content_type else "html"


def raw_path(source_id: str, manifest: dict) -> str:
    return str(RAW_DIR / manifest[source_id]["file"])


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def fetch_one(client: httpx.Client, src: Source, retries: int = 3) -> tuple[bytes, dict]:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.get(src.url)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            meta = {
                "url": src.url,
                "final_url": str(resp.url),
                "status": resp.status_code,
                "content_type": ctype,
                "bytes": len(resp.content),
                "sha256": hashlib.sha256(resp.content).hexdigest(),
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "file": f"{src.id}.{_ext(resp.content, ctype)}",
            }
            return resp.content, meta
        except (httpx.HTTPError, httpx.TransportError) as exc:
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {src.id} ({src.url}): {last_err}")


def fetch_all(refresh: bool = False) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    sources = load_sources()
    failures = []
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60
    ) as client:
        for src in sources:
            cached = manifest.get(src.id)
            if cached and (RAW_DIR / cached["file"]).exists() and not refresh:
                continue
            try:
                content, meta = fetch_one(client, src)
            except RuntimeError as exc:
                failures.append(str(exc))
                print(f"  ! {exc}")
                continue
            (RAW_DIR / meta["file"]).write_bytes(content)
            manifest[src.id] = meta
            print(f"  fetched {src.id:38} {meta['bytes']:>9,d} bytes")
            time.sleep(0.5)  # be polite to government servers
    # drop entries for sources that were removed from sources.yaml
    manifest = {k: v for k, v in manifest.items() if k in {s.id for s in sources}}
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    if failures:
        print(f"{len(failures)} source(s) failed; using {len(manifest)} snapshot(s).")
    return manifest


if __name__ == "__main__":
    fetch_all()
