"""Check candidate corpus URLs before adding them to sources.yaml.

Usage: python scripts/probe_urls.py urls.txt
Prints HTTP status, type, extracted-text length and a text preview for each URL.
TLS certificates are always verified; a URL that fails verification is simply not used.
"""

import concurrent.futures as cf
import sys

import httpx
import pymupdf
import trafilatura

UA = {"User-Agent": "Mozilla/5.0 (compatible; care-rag-research/0.1)"}


def probe(url: str) -> str:
    try:
        r = httpx.get(url, headers=UA, follow_redirects=True, timeout=40)
        ctype = r.headers.get("content-type", "")
        if "pdf" in ctype or r.content[:4] == b"%PDF":
            doc = pymupdf.open(stream=r.content, filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            kind = f"pdf{doc.page_count}p"
        else:
            text = trafilatura.extract(r.text) or ""
            kind = "html"
        preview = " ".join(text.split())[:90]
        return f"{r.status_code} {kind:7} {len(text):7d}  {url}\n      {preview!r}"
    except Exception as exc:  # report and move on; this is a probe
        return f"ERR {type(exc).__name__}: {str(exc)[:70]}  {url}"


def main(path: str) -> None:
    with open(path) as fh:
        urls = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    with cf.ThreadPoolExecutor(8) as pool:
        for line in pool.map(probe, urls):
            print(line)


if __name__ == "__main__":
    main(sys.argv[1])
