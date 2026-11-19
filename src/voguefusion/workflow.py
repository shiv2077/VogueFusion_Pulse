"""Bind request parameters onto a saved ComfyUI API-format workflow graph.

Parameters are written as values into the parsed JSON graph, never by
string templating, so user text cannot change the graph's structure. Nodes
are found by their role in the graph (the KSampler and whatever feeds its
positive and negative inputs), not by hard-coded node ids, so v1 and v2 bind
the same way.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

Graph = dict[str, dict[str, Any]]

# ComfyUI executing-node class -> job substate shown to the client.
SUBSTATE_BY_CLASS = {
    "CheckpointLoaderSimple": "loading_model",
    "LoraLoader": "loading_model",
    "IPAdapterUnifiedLoader": "loading_model",
    "LoadImage": "encoding_references",
    "IPAdapterEncoder": "encoding_references",
    "IPAdapterCombineEmbeds": "encoding_references",
    "IPAdapterEmbeds": "encoding_references",
    "CLIPTextEncode": "encoding_prompt",
    "KSampler": "sampling",
    "VAEDecode": "decoding",
    "SaveImage": "saving",
}


class WorkflowError(ValueError):
    pass


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_of(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def _only(graph: Graph, class_type: str) -> str:
    ids = [k for k, n in graph.items() if n["class_type"] == class_type]
    if len(ids) != 1:
        raise WorkflowError(f"expected one {class_type}, found {len(ids)}")
    return ids[0]


def reference_slots(graph: Graph) -> list[str]:
    """LoadImage node ids in a stable order (numeric id)."""
    return sorted((k for k, n in graph.items() if n["class_type"] == "LoadImage"), key=int)


def fill_slots(references: list[str], slots: int) -> list[str]:
    """Repeat references in order until every slot is filled (1..slots references allowed)."""
    if not 1 <= len(references) <= slots:
        raise WorkflowError(f"need 1..{slots} references, got {len(references)}")
    return [references[i % len(references)] for i in range(slots)]


def default_negative(graph: Graph) -> str:
    ks = graph[_only(graph, "KSampler")]
    return graph[ks["inputs"]["negative"][0]]["inputs"]["text"]


def default_batch(graph: Graph) -> int:
    return int(graph[_only(graph, "RepeatLatentBatch")]["inputs"]["amount"])


def bind(
    graph: Graph,
    *,
    prompt: str,
    negative: str,
    seed: int,
    batch: int,
    reference_files: list[str],
) -> Graph:
    """A new graph with the request's values set. reference_files are ComfyUI input filenames."""
    g = copy.deepcopy(graph)
    ks_id = _only(g, "KSampler")
    ks = g[ks_id]["inputs"]
    g[ks["positive"][0]]["inputs"]["text"] = prompt
    g[ks["negative"][0]]["inputs"]["text"] = negative
    ks["seed"] = seed
    g[_only(g, "RepeatLatentBatch")]["inputs"]["amount"] = batch
    slots = reference_slots(g)
    for node_id, name in zip(slots, fill_slots(reference_files, len(slots)), strict=True):
        g[node_id]["inputs"]["image"] = name
    return g


def output_node(graph: Graph) -> str:
    return _only(graph, "SaveImage")


def substate_for(graph: Graph, node_id: str | None) -> str | None:
    if node_id is None or node_id not in graph:
        return None
    return SUBSTATE_BY_CLASS.get(graph[node_id]["class_type"], "running")


def reference_filename(sha: str) -> str:
    """Name under which a reference blob is uploaded into ComfyUI's input folder."""
    return f"vf_{sha}.png"
