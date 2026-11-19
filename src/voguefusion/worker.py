"""Worker processes. One process per kind: `python -m voguefusion.worker generate|cost`.

The generate worker owns the GPU (through ComfyUI); the cost worker is CPU
only. They share the job table but never each other's jobs, so a long GPU
job cannot delay a quote and a burst of quotes cannot delay the GPU.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from typing import Any

from costing.catalogue import load_entries, to_dyes
from costing.extract import dominant_colours, load_pixels
from costing.quote import Fabric, quote
from voguefusion import config, logs
from voguefusion.comfy import ComfyClient, ComfyRejected
from voguefusion.jobstore import Job
from voguefusion.services import Services, build, load_workflow, model_hashes
from voguefusion.workflow import bind, output_node, reference_filename, substate_for

log = logging.getLogger("voguefusion.worker")

Handler = Callable[[Job, Callable[[str | None], None]], dict[str, Any]]


class LeaseLost(Exception):
    """Another worker owns this job now; stop without writing."""


class PermanentError(Exception):
    """Retrying cannot help (bad input, rejected graph)."""


def lease_keeper(svc: Services, job: Job, heartbeat_s: float) -> Callable[[str | None], None]:
    """Callback that renews the lease on substate change or once per heartbeat."""
    state = {"sub": None, "at": 0.0}

    def touch(substate: str | None) -> None:
        now = time.monotonic()
        changed = substate is not None and substate != state["sub"]
        if not changed and now - state["at"] < heartbeat_s:
            return
        if not svc.store.renew(job, substate if changed else None):
            raise LeaseLost(job.id)
        state["at"] = now
        if changed:
            state["sub"] = substate

    return touch


def generate_handler(svc: Services, comfy: ComfyClient) -> Handler:
    c = svc.cfg["comfyui"]
    heartbeat = svc.cfg["queue"]["heartbeat_s"]
    models = model_hashes(svc.cfg)

    def handle(job: Job, touch: Callable[[str | None], None]) -> dict[str, Any]:
        inp = job.input
        graph, wf_sha = load_workflow(svc.cfg, inp["workflow"])
        if wf_sha != inp["workflow_sha256"]:
            raise PermanentError("workflow file changed since the job was submitted")
        touch("uploading_references")
        names = []
        for sha in inp["references"]:
            comfy.upload_image(reference_filename(sha), svc.blobs.get(sha, verify=True))
            names.append(reference_filename(sha))
        bound = bind(
            graph,
            prompt=inp["prompt"],
            negative=inp["negative"],
            seed=inp["seed"],
            batch=inp["batch"],
            reference_files=names,
        )
        touch("queued_in_comfyui")
        t0 = time.monotonic()
        try:
            prompt_id = comfy.run(
                bound,
                lambda node: touch(substate_for(bound, node)),
                heartbeat_s=heartbeat,
                idle_timeout_s=c["ws_idle_timeout_s"],
                job_timeout_s=c["job_timeout_s"],
            )
        except ComfyRejected as e:
            raise PermanentError(f"ComfyUI rejected the graph: {e}") from e
        comfy_s = time.monotonic() - t0
        touch("fetching_outputs")
        artifacts = [
            svc.blobs.put(comfy.download(img))
            for img in comfy.outputs(prompt_id, output_node(bound))
        ]
        if not artifacts:
            raise PermanentError("ComfyUI finished without output images")
        used = {
            n["inputs"][k]
            for n in bound.values()
            for k in ("ckpt_name", "lora_name")
            if k in n["inputs"]
        }
        return {
            "artifacts": artifacts,
            "seed": inp["seed"],
            "workflow": inp["workflow"],
            "workflow_sha256": wf_sha,
            "models": {m: models.get(m) for m in sorted(used)},
            "comfy_prompt_id": prompt_id,
            "comfy_seconds": round(comfy_s, 3),
        }

    return handle


def cost_handler(svc: Services) -> Handler:
    c = svc.cfg["costing"]
    entries = load_entries(config.repo_path(c["catalogue_file"]))
    dyes = to_dyes(
        entries,
        c["reference_depth_g_per_kg"],
        c["swatch_reflectance_floor"],
        c["swatch_reflectance_ceiling"],
    )
    params = config.solver_params(svc.cfg)
    ex = c["extraction"]

    def handle(job: Job, touch: Callable[[str | None], None]) -> dict[str, Any]:
        inp = job.input
        if inp["solver_version"] != c["solver"]["version"]:
            raise PermanentError("solver version changed since the job was submitted")
        touch("extracting_colours")
        pixels = load_pixels(svc.blobs.get(inp["artifact"], verify=True), ex["max_side_px"])
        colours = dominant_colours(pixels, inp["n_colours"], ex["kmeans_n_init"], ex["kmeans_seed"])
        touch("solving")
        fabric = Fabric(
            inp["gsm"], inp["length_m"], inp["width_m"], tuple(c["substrates"][inp["substrate"]])
        )
        q = quote(
            colours,
            entries,
            dyes,
            fabric,
            params=params,
            tolerance=c["delta_e_tolerance"],
            reference_depth_g_per_kg=c["reference_depth_g_per_kg"],
            currency=c["currency"],
            solver_version=c["solver"]["version"],
        )
        return q.as_dict()

    return handle


def run_one(svc: Services, kind: str, worker_id: str, handler: Handler) -> bool:
    """Claim and process at most one job. Returns False if the queue was empty."""
    job = svc.store.claim(kind, worker_id)
    if job is None:
        return False
    fields = {"job_id": job.id, "kind": kind, "attempt": job.attempts, "worker": worker_id}
    log.info("job claimed", extra={"fields": fields})
    touch = lease_keeper(svc, job, svc.cfg["queue"]["heartbeat_s"])
    t0 = time.monotonic()
    try:
        result = handler(job, touch)
    except LeaseLost:
        log.warning("lease lost; abandoning job", extra={"fields": fields})
        return True
    except PermanentError as e:
        state = svc.store.fail(job, str(e), retryable=False)
        log.error(
            "job failed permanently", extra={"fields": {**fields, "error": str(e), "state": state}}
        )
        return True
    except Exception as e:  # noqa: BLE001 - any other failure is treated as transient
        state = svc.store.fail(job, f"{type(e).__name__}: {e}", retryable=True)
        log.warning(
            "job attempt failed",
            extra={"fields": {**fields, "error": f"{type(e).__name__}: {e}", "state": state}},
        )
        return True
    ok = svc.store.complete(job, result)
    log.info(
        "job succeeded" if ok else "lease lost before completion",
        extra={"fields": {**fields, "seconds": round(time.monotonic() - t0, 3)}},
    )
    return True


def run_forever(
    svc: Services, kind: str, handler: Handler, stop: threading.Event | None = None
) -> None:
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{kind}:{threading.get_ident()}"
    poll = svc.cfg["queue"]["poll_interval_s"]
    stop = stop or threading.Event()
    log.info("worker started", extra={"fields": {"kind": kind, "worker": worker_id}})
    while not stop.is_set():
        if not run_one(svc, kind, worker_id, handler):
            stop.wait(poll)


def make_handler(svc: Services, kind: str) -> Handler:
    if kind == "generate":
        c = svc.cfg["comfyui"]
        return generate_handler(svc, ComfyClient(c["url"], c["http_timeout_s"]))
    if kind == "cost":
        return cost_handler(svc)
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["generate", "cost"])
    args = ap.parse_args()
    logs.setup()
    svc = build(config.load())
    run_forever(svc, args.kind, make_handler(svc, args.kind))


if __name__ == "__main__":
    main()
