"""Shared wiring for the API and workers: stores, workflows, and job-input normalisation.

A job's input_key is the SHA-256 of its normalised input. Two requests with
the same key would produce the same output, so the job store uses it to
coalesce in-flight duplicates and to return cached results.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voguefusion import config
from voguefusion.blobstore import BlobStore
from voguefusion.jobstore import JobStore, KindLimits
from voguefusion.workflow import Graph, default_negative, sha256_of


@dataclass(frozen=True)
class Services:
    cfg: dict[str, Any]
    data_dir: Path
    store: JobStore
    blobs: BlobStore


def kind_limits(cfg: dict[str, Any]) -> dict[str, KindLimits]:
    return {
        kind: KindLimits(k["max_depth"], k["max_attempts"], k["lease_s"], k["retry_backoff_s"])
        for kind, k in cfg["queue"]["kinds"].items()
    }


def build(cfg: dict[str, Any]) -> Services:
    data = config.data_dir(cfg)
    store = JobStore(
        data / cfg["queue"]["db_file"], kind_limits(cfg), cfg["queue"]["sqlite_busy_timeout_ms"]
    )
    return Services(cfg, data, store, BlobStore(data / "blobs"))


def load_workflow(cfg: dict[str, Any], name: str) -> tuple[Graph, str]:
    """(graph, sha256 of the file bytes) for a configured workflow name."""
    entry = cfg["generate"]["workflows"][name]
    raw = (config.repo_path(cfg["comfyui"]["workflows_dir"]) / entry["file"]).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def model_hashes(cfg: dict[str, Any]) -> dict[str, str]:
    """Model file -> expected sha256, from the manifest the downloader verifies against."""
    manifest = json.loads(config.repo_path(cfg["comfyui"]["models_manifest"]).read_text())
    return {Path(m["dest"]).name: m["sha256"] for m in manifest["models"]}


def generate_input(
    cfg: dict[str, Any],
    *,
    workflow: str,
    prompt: str,
    negative: str | None,
    seed: int,
    batch: int,
    references: list[str],
) -> tuple[dict[str, Any], str]:
    """Normalised generate-job input and its input_key."""
    graph, wf_sha = load_workflow(cfg, workflow)
    payload = {
        "workflow": workflow,
        "workflow_sha256": wf_sha,
        "prompt": prompt,
        "negative": default_negative(graph) if negative is None else negative,
        "seed": seed,
        "batch": batch,
        "references": references,
    }
    return payload, sha256_of(payload)


def catalogue_sha(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(
        config.repo_path(cfg["costing"]["catalogue_file"]).read_bytes()
    ).hexdigest()


def cost_input(
    cfg: dict[str, Any],
    *,
    artifact: str,
    gsm: float,
    length_m: float,
    width_m: float,
    substrate: str,
    n_colours: int,
) -> tuple[dict[str, Any], str]:
    """Normalised cost-job input and its input_key.

    The key covers everything the quote depends on, including the solver
    version, the catalogue file and the costing config, so changing any of
    them invalidates cached quotes.
    """
    c = cfg["costing"]
    payload = {
        "artifact": artifact,
        "gsm": gsm,
        "length_m": length_m,
        "width_m": width_m,
        "substrate": substrate,
        "n_colours": n_colours,
        "solver_version": c["solver"]["version"],
        "catalogue_sha256": catalogue_sha(cfg),
        "costing_config_sha256": sha256_of(c),
    }
    return payload, sha256_of(payload)
