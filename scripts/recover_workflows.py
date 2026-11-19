"""Recover the ComfyUI workflows from PNG metadata into comfyui/workflows/.

v1 is the exact API graph embedded in dataset/initial_dataset/generated/image_9.png.
v2 is v1 plus the LoRA loader, batch of 3 and the long negative prompt, as shown
in assets/images/workflow.png and assets/images/frontend.png.

Idempotent: rewrites the same files with the same content.
Run: python scripts/recover_workflows.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PNG = ROOT / "dataset/initial_dataset/generated/image_9.png"
OUT = ROOT / "comfyui/workflows"

LORA_NODE_ID = "40"
V2_NEGATIVE = (
    "NSFW, FastNegativeV2 negative_hand-neg, noise, blur, NSFW, nude, (worst quality, "
    "low quality, normal quality), (deformed iris, deformed pupils), error, cropped, "
    "blurry,  crooked teeth, username,  multiple views, mutation, bad anatomy, bad "
    "proportions, extra limbs, (cloned face), disfigured, gross proportions, malformed "
    "limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many "
    "fingers, (long neck)"
)


def load_embedded(png: Path) -> tuple[dict, dict]:
    """Return (api_graph, ui_graph) from a ComfyUI-written PNG."""
    info = Image.open(png).info
    api = json.loads(info["prompt"])
    for node in api.values():
        node.pop("is_changed", None)
    return api, json.loads(info["workflow"])


def make_v2(v1: dict) -> dict:
    """Add a LoraLoader between the checkpoint and its consumers, batch 3, long negative."""
    g = copy.deepcopy(v1)
    g[LORA_NODE_ID] = {
        "class_type": "LoraLoader",
        "inputs": {
            "lora_name": "VogueFusion.safetensors",
            "strength_model": 1.0,
            "strength_clip": 1.0,
            "model": ["4", 0],
            "clip": ["4", 1],
        },
    }
    g["20"]["inputs"]["model"] = [LORA_NODE_ID, 0]
    g["6"]["inputs"]["clip"] = [LORA_NODE_ID, 1]
    g["7"]["inputs"]["clip"] = [LORA_NODE_ID, 1]
    g["7"]["inputs"]["text"] = V2_NEGATIVE
    g["10"]["inputs"]["amount"] = 3
    return g


def dump(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    api, ui = load_embedded(SOURCE_PNG)
    dump(api, OUT / "v1.api.json")
    dump(ui, OUT / "v1.ui.json")
    dump(make_v2(api), OUT / "v2.api.json")
    print("wrote", *sorted(p.name for p in OUT.glob("*.json")))


if __name__ == "__main__":
    main()
