"""HTTP API. Run with `python -m voguefusion.api`.

The API never talks to the GPU. It validates input, writes jobs to the job
store and serves results; workers do the work. Generation and costing are
asynchronous: submit returns 202 with a job id and a Location to poll.
"""

import hashlib
import io
import json
import math
import secrets
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from costing.catalogue import load_entries
from voguefusion import config, logs
from voguefusion.blobstore import BlobNotFound
from voguefusion.comfy import ComfyClient
from voguefusion.jobstore import IdempotencyConflict, QueueFull
from voguefusion.services import Services, build, cost_input, generate_input, load_workflow
from voguefusion.workflow import default_batch, default_negative, reference_slots

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}


def retry_after_seconds(
    depth: int, service_s: float, workers: int, min_s: float, max_s: float
) -> int:
    """Expected time for the current backlog to drain, clamped: ceil(depth * service / workers)."""
    return int(min(max_s, max(min_s, math.ceil(depth * service_s / max(workers, 1)))))


def default_seed(idempotency_key: str | None, seed_max: int) -> int:
    """Seed for a request that did not give one.

    Random, except under an Idempotency-Key, where it is derived from the key
    so that a retried request normalises to the same input as the original.
    """
    if idempotency_key is None:
        return secrets.randbelow(seed_max + 1)
    digest = hashlib.sha256(idempotency_key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % (seed_max + 1)


def normalise_image(data: bytes, min_side: int, max_side: int) -> tuple[bytes, int, int]:
    """Decode untrusted image bytes and re-encode as plain RGB PNG.

    Re-encoding drops metadata and anything appended to the file, so what is
    stored is exactly the pixels that were checked.
    """
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError) as e:
        raise HTTPException(415, f"not a readable image: {type(e).__name__}") from e
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(415, f"image format {fmt} not allowed; use {sorted(ALLOWED_FORMATS)}")
    w, h = img.size
    if min(w, h) < min_side or max(w, h) > max_side:
        raise HTTPException(422, f"image is {w}x{h}; each side must be {min_side}..{max_side} px")
    out = io.BytesIO()
    img.convert("RGB").save(out, format="PNG")
    return out.getvalue(), w, h


def create_app(svc: Services, comfy: ComfyClient | None = None) -> FastAPI:
    cfg = svc.cfg
    lim = cfg["limits"]
    workflows = cfg["generate"]["workflows"]
    Image.MAX_IMAGE_PIXELS = lim["image_max_side"] ** 2

    class DesignRequest(BaseModel):
        workflow: str = cfg["generate"]["default_workflow"]
        prompt: str = Field(min_length=1, max_length=lim["prompt_max_chars"])
        negative: str | None = Field(default=None, max_length=lim["negative_max_chars"])
        seed: int | None = Field(default=None, ge=0, le=lim["seed_max"])
        batch: int = Field(default=1, ge=1, le=lim["batch_max"])
        references: list[str] = Field(
            min_length=lim["references_min"], max_length=lim["references_max"]
        )

    class CostRequest(BaseModel):
        artifact: str = Field(pattern=r"^[0-9a-f]{64}$")
        gsm: float = Field(default=cfg["costing"]["fabric"]["default_gsm"], gt=0, le=1000)
        length_m: float = Field(default=cfg["costing"]["fabric"]["default_length_m"], gt=0, le=1000)
        width_m: float = Field(default=cfg["costing"]["fabric"]["default_width_m"], gt=0, le=10)
        substrate: str = cfg["costing"]["default_substrate"]
        n_colours: int = Field(default=cfg["costing"]["extraction"]["n_colours"], ge=1, le=12)

    app = FastAPI(title="VogueFusion", version="1")

    def submit(
        kind: str, payload: dict[str, Any], key: str, idem: str | None, path: str
    ) -> JSONResponse:
        try:
            job, created = svc.store.submit(kind, payload, key, idempotency_key=idem)
        except QueueFull as e:
            stats = svc.store.stats(kind, cfg["queue"]["metrics_window"])
            kcfg = cfg["queue"]["kinds"][kind]
            service = stats["service_s"]["p50"] or kcfg["default_service_s"]
            ra = retry_after_seconds(
                e.depth, service, kcfg["workers"], **cfg["queue"]["retry_after"]
            )
            return JSONResponse(
                {"detail": str(e), "depth": e.depth, "retry_after_s": ra},
                status_code=429,
                headers={"Retry-After": str(ra)},
            )
        except IdempotencyConflict as e:
            raise HTTPException(409, "Idempotency-Key reused with a different request") from e
        return JSONResponse(
            {**job.public(), "deduplicated": not created},
            status_code=202 if created else 200,
            headers={"Location": f"{path}/{job.id}"},
        )

    def get_job(kind: str, job_id: str) -> dict[str, Any]:
        job = svc.store.get(job_id)
        if job is None or job.kind != kind:
            raise HTTPException(404, "no such job")
        return job.public()

    @app.post("/v1/references", status_code=201)
    async def upload_reference(file: UploadFile) -> dict[str, Any]:
        data = await file.read(lim["upload_max_bytes"] + 1)
        if len(data) > lim["upload_max_bytes"]:
            raise HTTPException(413, f"upload exceeds {lim['upload_max_bytes']} bytes")
        png, w, h = normalise_image(data, lim["image_min_side"], lim["image_max_side"])
        return {"sha256": svc.blobs.put(png), "width": w, "height": h}

    @app.get("/v1/workflows")
    def list_workflows() -> dict[str, Any]:
        """Workflows with the defaults saved in each graph."""
        out = {}
        for name, entry in workflows.items():
            graph, sha = load_workflow(cfg, name)
            out[name] = {
                **entry,
                "sha256": sha,
                "default_negative": default_negative(graph),
                "default_batch": default_batch(graph),
                "reference_slots": len(reference_slots(graph)),
            }
        return {"default": cfg["generate"]["default_workflow"], "workflows": out}

    @app.post("/v1/designs")
    def submit_design(
        req: DesignRequest, idempotency_key: Annotated[str | None, Header(max_length=200)] = None
    ) -> JSONResponse:
        if req.workflow not in workflows:
            raise HTTPException(422, f"unknown workflow {req.workflow!r}")
        missing = [r for r in req.references if not svc.blobs.exists(r)]
        if missing:
            raise HTTPException(422, f"unknown reference(s): {missing}; upload to /v1/references")
        seed = req.seed if req.seed is not None else default_seed(idempotency_key, lim["seed_max"])
        payload, key = generate_input(
            cfg,
            workflow=req.workflow,
            prompt=req.prompt,
            negative=req.negative,
            seed=seed,
            batch=req.batch,
            references=req.references,
        )
        return submit("generate", payload, key, idempotency_key, "/v1/designs")

    @app.get("/v1/designs/{job_id}")
    def design_status(job_id: str) -> dict[str, Any]:
        return get_job("generate", job_id)

    @app.post("/v1/costs")
    def submit_cost(
        req: CostRequest, idempotency_key: Annotated[str | None, Header(max_length=200)] = None
    ) -> JSONResponse:
        if req.substrate not in cfg["costing"]["substrates"]:
            raise HTTPException(422, f"unknown substrate {req.substrate!r}")
        if not svc.blobs.exists(req.artifact):
            raise HTTPException(422, "unknown artifact")
        payload, key = cost_input(cfg, **req.model_dump())
        return submit("cost", payload, key, idempotency_key, "/v1/costs")

    @app.get("/v1/costs/{job_id}")
    def cost_status(job_id: str) -> dict[str, Any]:
        return get_job("cost", job_id)

    @app.get("/v1/artifacts/{sha}")
    def artifact(sha: str, request: Request) -> Response:
        etag = f'"{sha}"'
        if request.headers.get("if-none-match") == etag and svc.blobs.exists(sha):
            return Response(status_code=304, headers={"ETag": etag})
        try:
            data = svc.blobs.get(sha)
        except BlobNotFound as e:
            raise HTTPException(404, "no such artifact") from e
        return Response(
            data,
            media_type="image/png",
            headers={"ETag": etag, "Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get("/v1/suggestions")
    def suggestions() -> dict[str, Any]:
        """Prompt keywords from the latest offline trend build (scripts/build_trends.py).

        Read from a file the batch job writes; nothing here waits on the batch job.
        """
        path = svc.data_dir / cfg["trends"]["store_dir"] / "current.json"
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return {"version": None, "keywords": [], "label": "no trend build yet"}

    @app.get("/v1/metrics")
    def metrics() -> dict[str, Any]:
        window = cfg["queue"]["metrics_window"]
        return {kind: svc.store.stats(kind, window) for kind in cfg["queue"]["kinds"]}

    @app.get("/v1/costing")
    def costing() -> dict[str, Any]:
        """What a costing client needs: dye catalogue, fabric defaults, substrates, tolerance."""
        c = cfg["costing"]
        entries = load_entries(config.repo_path(c["catalogue_file"]))
        return {
            "currency": c["currency"],
            "delta_e_tolerance": c["delta_e_tolerance"],
            "default_n_colours": c["extraction"]["n_colours"],
            "fabric": c["fabric"],
            "substrates": list(c["substrates"]),
            "default_substrate": c["default_substrate"],
            "dyes": [
                {"name": e.name, "rgb": list(e.srgb_255), "price_per_kg": e.price_per_kg}
                for e in entries
            ],
        }

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        svc.store.depth("generate")  # raises if the database is unusable
        return {"status": "ok", "comfyui_reachable": comfy.reachable() if comfy else None}

    # The web app: static files calling this API from the same origin. Mounted
    # last, so every API route above takes precedence over it.
    app.mount(
        "/", StaticFiles(directory=config.repo_path(cfg["ui"]["web_dir"]), html=True), name="web"
    )
    return app


def main() -> None:
    import uvicorn

    logs.setup()
    cfg = config.load()
    svc = build(cfg)
    comfy = ComfyClient(cfg["comfyui"]["url"], cfg["comfyui"]["http_timeout_s"])
    uvicorn.run(
        create_app(svc, comfy), host=cfg["api"]["host"], port=cfg["api"]["port"], log_config=None
    )


if __name__ == "__main__":
    main()
