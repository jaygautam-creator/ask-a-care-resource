"""Load and validate the corpus definition in sources.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from care_rag.config import SOURCES_FILE


@dataclass(frozen=True)
class Source:
    id: str
    url: str
    title: str
    category: str
    lang: str
    publisher: str


def load_sources(path: Path = SOURCES_FILE) -> list[Source]:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)["sources"]
    sources = [Source(**item) for item in raw]
    ids = [s.id for s in sources]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate source ids in {path}: {sorted(dupes)}")
    return sources
