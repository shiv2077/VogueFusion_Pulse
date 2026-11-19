"""Offline trend pipeline: keyword statistics -> versioned prompt-conditioning store.

Runs as a batch job (scripts/build_trends.py), never in the request path.
The API only reads the file this writes (ADR-009).

Two inputs are supported:

- a caption corpus (one comma-separated tag list per image), which gives
  keyword frequencies but no time axis;
- a seasonal CSV (season,keywords), which additionally gives a trend lift:
  how much more often a keyword appears in the most recent seasons than in
  the seasons before them.
"""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from voguefusion.workflow import sha256_of

SEASON_ORDER = {"winter": 0, "spring": 1, "summer": 2, "fall": 3}
SEASON_ALIASES = {"autumn": "fall"}
SEASON_RE = re.compile(r"^\s*([A-Za-z]+)\s+(\d{4})\s*$")


def parse_tags(text: str) -> list[str]:
    """Comma-separated tags, stripped, lower-cased, de-duplicated in order."""
    out: list[str] = []
    for t in text.split(","):
        t = t.strip().lower()
        if t and t not in out:
            out.append(t)
    return out


def clean(tags: Iterable[str], exclude: set[str]) -> list[str]:
    return [t for t in tags if t not in exclude]


def document_frequency(docs: Iterable[list[str]]) -> Counter[str]:
    """Number of documents each tag appears in."""
    c: Counter[str] = Counter()
    for d in docs:
        c.update(set(d))
    return c


def season_key(label: str) -> tuple[int, int]:
    """'Autumn 2021' -> (2021, 3). Fall and Autumn are the same season."""
    m = SEASON_RE.match(label)
    if not m:
        raise ValueError(f"not a season label: {label!r}")
    name = SEASON_ALIASES.get(m.group(1).lower(), m.group(1).lower())
    if name not in SEASON_ORDER:
        raise ValueError(f"unknown season: {label!r}")
    return int(m.group(2)), SEASON_ORDER[name]


def trend_lift(
    records: list[tuple[str, list[str]]], recent_seasons: int, min_support: int
) -> list[dict[str, Any]]:
    """Per keyword: share of items in the most recent seasons vs. all earlier seasons.

    lift = recent_share / baseline_share; a keyword absent from the baseline
    gets baseline_share 0 and lift None (new, not measurable as a ratio).
    Keywords seen fewer than min_support times overall are dropped.
    """
    keys = sorted({season_key(s) for s, _ in records})
    if len(keys) <= recent_seasons:
        raise ValueError("need more seasons than recent_seasons to form a baseline")
    recent = set(keys[-recent_seasons:])
    rec_docs = [t for s, t in records if season_key(s) in recent]
    base_docs = [t for s, t in records if season_key(s) not in recent]
    rec_df, base_df = document_frequency(rec_docs), document_frequency(base_docs)
    rows = []
    for kw in sorted(set(rec_df) | set(base_df)):
        if rec_df[kw] + base_df[kw] < min_support:
            continue
        r = rec_df[kw] / len(rec_docs)
        b = base_df[kw] / len(base_docs)
        rows.append(
            {
                "keyword": kw,
                "recent_share": round(r, 4),
                "baseline_share": round(b, 4),
                "lift": None if b == 0 else round(r / b, 3),
            }
        )
    rows.sort(
        key=lambda x: (x["lift"] is not None, x["lift"] or 0, x["recent_share"]), reverse=True
    )
    return rows


def read_captions(directory: Path) -> list[list[str]]:
    return [parse_tags(p.read_text()) for p in sorted(Path(directory).glob("*.txt"))]


def read_seasonal_csv(path: Path) -> list[tuple[str, list[str]]]:
    with open(path, newline="") as f:
        return [(row["season"], parse_tags(row["keywords"])) for row in csv.DictReader(f)]


def build_store(
    *,
    source: str,
    label: str,
    docs: list[list[str]],
    seasonal: list[tuple[str, list[str]]] | None,
    exclude: set[str],
    top_n: int,
    recent_seasons: int,
    min_support: int,
    caveats: list[str],
) -> dict[str, Any]:
    """The store document. version is a hash of everything except itself."""
    cleaned = [clean(d, exclude) for d in docs]
    df = document_frequency(cleaned)
    keywords = [k for k, _ in sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]
    lift = None
    if seasonal:
        lift = trend_lift(
            [(s, clean(t, exclude)) for s, t in seasonal], recent_seasons, min_support
        )
        keywords = [r["keyword"] for r in lift[:top_n]]
    body = {
        "source": source,
        "label": label,
        "documents": len(docs),
        "excluded_tags": sorted(exclude),
        "keywords": keywords,
        "frequency": {k: df[k] for k in keywords if k in df},
        "lift": lift,
        "caveats": caveats,
    }
    return {"version": sha256_of(body)[:16], **body}


def write_store(directory: Path, store: dict[str, Any]) -> bool:
    """Write <version>.json and point current.json at it, atomically.

    Returns False (and writes nothing) if current.json already has this version.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    current = directory / "current.json"
    if current.exists() and json.loads(current.read_text()).get("version") == store["version"]:
        return False
    text = json.dumps(store, indent=2, sort_keys=True)
    for target in (directory / f"{store['version']}.json", current):
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-")
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    return True
