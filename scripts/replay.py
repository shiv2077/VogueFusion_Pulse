"""Replay a committed generated PNG through ComfyUI and compare the result.

Each PNG in dataset/initial_dataset/generated/ embeds the exact API graph and
seed that produced it. This script uploads the four reference images under
the filenames the graph expects (mapped by SHA-256 to files in this repo),
queues the embedded graph unchanged, and compares the new image to the
committed one two ways:

- mean per-pixel CIEDE2000, which must be lower against the original than
  against every other committed output (different prompts and seeds);
- area-weighted CIEDE2000 between the two images' dominant 5-colour
  palettes, which must be within --palette-threshold.

Bit-identical output is not expected across GPUs, drivers and ComfyUI
versions; motif placement shifts, so per-pixel distance alone overstates the
difference between two renderings of the same design.

Run: PYTHONPATH=src python scripts/replay.py [image_9.png ...]
Exits non-zero if any replay fails. Results go to $VOGUEFUSION_DATA/replay/.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from costing.color import perceptual as p
from costing.extract import dominant_colours, load_pixels
from voguefusion import config
from voguefusion.comfy import ComfyClient
from voguefusion.workflow import output_node

ROOT = config.REPO_ROOT
GENERATED = ROOT / "dataset/initial_dataset/generated"
SEARCH = [ROOT / "dataset/initial_dataset/Saree", ROOT / "dataset/training_data"]


def files_by_sha() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in SEARCH:
        for f in sorted(d.glob("*.jp*g")):
            out.setdefault(hashlib.sha256(f.read_bytes()).hexdigest(), f)
    return out


def embedded(png: Path) -> tuple[dict, dict[str, str]]:
    """(graph without is_changed, LoadImage filename -> sha256 of its content)."""
    graph = json.loads(Image.open(png).info["prompt"])
    refs = {}
    for node in graph.values():
        if node["class_type"] == "LoadImage":
            refs[node["inputs"]["image"]] = node.pop("is_changed")[0]
        node.pop("is_changed", None)
    return graph, refs


def mean_delta_e(a: Image.Image, b: Image.Image) -> float:
    """Mean CIEDE2000 over pixels, after resizing b to a's size."""
    x = np.asarray(a.convert("RGB"), dtype=np.float64) / 255.0
    y = np.asarray(b.convert("RGB").resize(a.size), dtype=np.float64) / 255.0
    return float(np.mean(p.delta_e_2000(p.srgb_to_lab(x), p.srgb_to_lab(y))))


def palette_delta_e(a: Image.Image, b: Image.Image, n: int = 5) -> float:
    """Area-weighted CIEDE2000 from each dominant colour of a to its nearest in b."""

    def palette(img: Image.Image) -> list:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return dominant_colours(load_pixels(buf.getvalue(), 256), n, 4, 0)

    pa, pb = palette(a), palette(b)
    lab_b = p.srgb_to_lab(np.array([c.srgb for c in pb]))
    return float(
        sum(c.fraction * p.delta_e_2000(p.srgb_to_lab(np.array(c.srgb)), lab_b).min() for c in pa)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", default=["image_9.png"])
    ap.add_argument("--palette-threshold", type=float, default=5.0)
    args = ap.parse_args()
    cfg = config.load()
    out_dir = config.data_dir(cfg) / "replay"
    out_dir.mkdir(exist_ok=True)
    comfy = ComfyClient(cfg["comfyui"]["url"], cfg["comfyui"]["http_timeout_s"])
    if not comfy.reachable():
        print(f"ComfyUI not reachable at {cfg['comfyui']['url']}", file=sys.stderr)
        return 2
    by_sha = files_by_sha()
    others = sorted(GENERATED.glob("*.png"))
    failed = False
    report = []
    for name in args.images:
        src = GENERATED / name
        graph, refs = embedded(src)
        for filename, sha in refs.items():
            comfy.upload_image(filename, by_sha[sha].read_bytes())
        t0 = time.monotonic()
        nodes: list[tuple[float, str | None]] = []
        prompt_id = comfy.run(
            graph,
            lambda n, _n=nodes, _t=t0: _n.append((time.monotonic() - _t, n)) if n else None,
            heartbeat_s=cfg["queue"]["heartbeat_s"],
            idle_timeout_s=cfg["comfyui"]["ws_idle_timeout_s"],
            job_timeout_s=cfg["comfyui"]["job_timeout_s"],
        )
        seconds = time.monotonic() - t0
        imgs = comfy.outputs(prompt_id, output_node(graph))
        new = Image.open(io.BytesIO(comfy.download(imgs[0])))
        new.save(out_dir / f"replay_{name}")
        original = Image.open(src)
        de_self = mean_delta_e(original, new)
        de_others = {o.name: round(mean_delta_e(Image.open(o), new), 2) for o in others if o != src}
        pal_self = palette_delta_e(original, new)
        pal_others = min(palette_delta_e(Image.open(o), new) for o in others if o != src)
        row = {
            "image": name,
            "seed": graph["3"]["inputs"]["seed"],
            "seconds": round(seconds, 2),
            "mean_delta_e_vs_original": round(de_self, 2),
            "mean_delta_e_vs_other_outputs_min": min(de_others.values()),
            "mean_delta_e_vs_other_outputs": de_others,
            "palette_delta_e_vs_original": round(pal_self, 2),
            "palette_delta_e_vs_other_outputs_min": round(pal_others, 2),
            "node_timeline_s": [(round(t, 2), n) for t, n in nodes],
            "pass": de_self < min(de_others.values()) and pal_self <= args.palette_threshold,
        }
        failed |= not row["pass"]
        report.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "node_timeline_s"}))
    (out_dir / "replay_report.json").write_text(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
