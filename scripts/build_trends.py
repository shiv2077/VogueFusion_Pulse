"""Batch job: build the prompt-conditioning store the API serves at /v1/suggestions.

Default input is the LoRA training caption corpus (dataset/training_data),
which has no time axis, so the store holds domain keywords rather than a
trend signal. Pass --seasonal CSV (columns: season,keywords) to compute
trend lift from a dated dataset instead.

Idempotent: rebuilding from unchanged inputs is a no-op.

Run: PYTHONPATH=src python scripts/build_trends.py [--seasonal path.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from voguefusion import config
from voguefusion.trends import build_store, read_captions, read_seasonal_csv, write_store


def main() -> None:
    cfg = config.load()
    t = cfg["trends"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default=t["captions_dir"])
    ap.add_argument("--seasonal", default=None)
    args = ap.parse_args()
    docs = read_captions(config.repo_path(args.captions))
    seasonal = read_seasonal_csv(Path(args.seasonal)) if args.seasonal else None
    if seasonal:
        source, label = f"seasonal:{args.seasonal}", "trending: lift of recent seasons"
        caveats = t["caveats"]["seasonal"]
    else:
        source, label = f"captions:{args.captions}", "domain keywords (no time axis)"
        caveats = t["caveats"]["captions"]
    store = build_store(
        source=source,
        label=label,
        docs=docs,
        seasonal=seasonal,
        exclude=set(t["exclude_tags"]),
        top_n=t["top_n"],
        recent_seasons=t["recent_seasons"],
        min_support=t["min_support"],
        caveats=caveats,
    )
    out = config.data_dir(cfg) / t["store_dir"]
    changed = write_store(out, store)
    print(("wrote" if changed else "unchanged"), out / "current.json", "version", store["version"])
    print("keywords:", ", ".join(store["keywords"]))


if __name__ == "__main__":
    main()
