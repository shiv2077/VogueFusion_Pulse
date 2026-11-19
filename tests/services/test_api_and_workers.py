"""API validation and worker behaviour end to end against the fake ComfyUI."""

import os
import signal
import subprocess
import sys
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.services.conftest import make_config, png_bytes, two_tone_png, write_config
from voguefusion import config
from voguefusion.api import create_app, default_seed, retry_after_seconds
from voguefusion.comfy import ComfyClient
from voguefusion.services import build
from voguefusion.worker import cost_handler, generate_handler, make_handler, run_forever, run_one


def client_for(svc, comfy_url=None):
    comfy = ComfyClient(comfy_url, 2) if comfy_url else None
    return TestClient(create_app(svc, comfy))


def upload(api, data=None):
    r = api.post("/v1/references", files={"file": ("r.png", data or png_bytes(), "image/png")})
    assert r.status_code == 201, r.text
    return r.json()["sha256"]


def design(api, ref, **kw):
    body = {"prompt": "floral border", "references": [ref], "seed": 5, **kw}
    return api.post("/v1/designs", json=body)


def gen_handler(svc):
    c = svc.cfg["comfyui"]
    return generate_handler(svc, ComfyClient(c["url"], c["http_timeout_s"]))


def drain(svc, kind, handler, max_rounds=20):
    """Run the worker until the queue is empty, sleeping across retry backoffs."""
    for _ in range(max_rounds):
        if not run_one(svc, kind, "test-worker", handler):
            if svc.store.depth(kind) == 0:
                return
            time.sleep(svc.cfg["queue"]["kinds"][kind]["retry_backoff_s"] * 3)


def set_mode(fake, mode, delay_s=0.01):
    httpx.post(f"{fake.url}/_fake/mode", json={"mode": mode, "delay_s": delay_s})


# -- API validation ---------------------------------------------------------------


def test_upload_rejects_non_images_and_bad_sizes(svc):
    api = client_for(svc)
    assert api.post("/v1/references", files={"file": ("x.png", b"not an image")}).status_code == 415
    tiny = api.post("/v1/references", files={"file": ("t.png", png_bytes(size=(10, 10)))})
    assert tiny.status_code == 422


def test_upload_size_limit(tmp_path, fake_comfy):
    cfg = make_config(tmp_path, fake_comfy.url)
    cfg["limits"]["upload_max_bytes"] = 1000
    api = client_for(build(cfg))
    big = png_bytes(size=(1000, 1000))
    assert len(big) > 1000
    assert api.post("/v1/references", files={"file": ("b.png", big)}).status_code == 413


def test_upload_is_normalised_so_trailing_bytes_are_dropped(svc):
    api = client_for(svc)
    a = upload(api, png_bytes())
    b = upload(api, png_bytes() + b"<script>trailing payload</script>")
    assert a == b


def test_design_validation(svc):
    api = client_for(svc)
    ref = upload(api)
    assert design(api, "0" * 64).status_code == 422  # unknown reference
    assert design(api, ref, workflow="v9").status_code == 422
    assert design(api, ref, prompt="x" * 10_000).status_code == 422
    assert design(api, ref, batch=99).status_code == 422
    assert design(api, ref, references=[ref] * 5).status_code == 422


def test_design_submit_dedupe_and_idempotency(svc):
    api = client_for(svc)
    ref = upload(api)
    first = design(api, ref)
    assert first.status_code == 202
    assert first.headers["location"] == f"/v1/designs/{first.json()['id']}"
    again = design(api, ref)
    assert again.status_code == 200 and again.json()["deduplicated"]
    assert again.json()["id"] == first.json()["id"]
    k = {"Idempotency-Key": "abc"}
    a = api.post("/v1/designs", json={"prompt": "p1", "references": [ref]}, headers=k)
    b = api.post("/v1/designs", json={"prompt": "p1", "references": [ref]}, headers=k)
    assert a.status_code == 202 and b.status_code == 200 and a.json()["id"] == b.json()["id"]
    c = api.post("/v1/designs", json={"prompt": "other", "references": [ref]}, headers=k)
    assert c.status_code == 409


def test_server_generates_seed_when_omitted(svc):
    api = client_for(svc)
    ref = upload(api)
    a = api.post("/v1/designs", json={"prompt": "p", "references": [ref]}).json()
    b = api.post("/v1/designs", json={"prompt": "p", "references": [ref]}).json()
    assert a["id"] != b["id"] and a["input"]["seed"] != b["input"]["seed"]


def test_backpressure_returns_429_with_retry_after(svc):
    api = client_for(svc)
    ref = upload(api)
    limit = svc.cfg["queue"]["kinds"]["generate"]["max_depth"]
    for seed in range(limit):
        assert design(api, ref, seed=seed).status_code == 202
    r = design(api, ref, seed=999)
    assert r.status_code == 429
    expected = retry_after_seconds(limit, 60, 1, 1, 900)
    assert int(r.headers["retry-after"]) == expected == r.json()["retry_after_s"]


def test_default_seed_is_random_without_key_and_stable_with_one():
    assert default_seed("k1", 2**64 - 1) == default_seed("k1", 2**64 - 1)
    assert default_seed("k1", 2**64 - 1) != default_seed("k2", 2**64 - 1)
    assert len({default_seed(None, 2**64 - 1) for _ in range(5)}) == 5


def test_retry_after_closed_form():
    assert retry_after_seconds(8, 60, 1, 1, 900) == 480
    assert retry_after_seconds(8, 60, 2, 1, 900) == 240
    assert retry_after_seconds(0, 60, 1, 1, 900) == 1
    assert retry_after_seconds(100, 60, 1, 1, 900) == 900


def test_unknown_and_wrong_kind_jobs_are_404(svc):
    api = client_for(svc)
    ref = upload(api)
    job_id = design(api, ref).json()["id"]
    assert api.get("/v1/designs/nope").status_code == 404
    assert api.get(f"/v1/costs/{job_id}").status_code == 404


def test_metrics_and_health(svc, fake_comfy):
    api = client_for(svc, fake_comfy.url)
    m = api.get("/v1/metrics").json()
    assert set(m) == {"generate", "cost"} and m["generate"]["depth"] == 0
    assert api.get("/healthz").json() == {"status": "ok", "comfyui_reachable": True}


# -- workers --------------------------------------------------------------------


def test_generate_job_end_to_end(svc, fake_comfy):
    api = client_for(svc)
    refs = [upload(api, png_bytes((i * 50, 20, 20))) for i in range(2)]
    job_id = design(api, refs[0], references=refs, batch=2).json()["id"]
    drain(svc, "generate", gen_handler(svc))
    job = api.get(f"/v1/designs/{job_id}").json()
    assert job["state"] == "succeeded", job
    res = job["result"]
    assert len(res["artifacts"]) == 2 and res["seed"] == 5
    ckpt_sha = "832e7bb2302c3cd67c818ca4fe9dcbedf696d3070dab5463127263ec4db9899f"
    assert res["models"] == {"dreamshaperXL_sfwLightningDPMSDE.safetensors": ckpt_sha}
    art = api.get(f"/v1/artifacts/{res['artifacts'][0]}")
    assert art.status_code == 200 and art.headers["etag"] == f'"{res["artifacts"][0]}"'
    cached = api.get(
        f"/v1/artifacts/{res['artifacts'][0]}", headers={"If-None-Match": art.headers["etag"]}
    )
    assert cached.status_code == 304
    uploads = httpx.get(f"{fake_comfy.url}/_fake/stats").json()["uploads"]
    assert uploads == sorted(f"vf_{r}.png" for r in refs)


def test_cached_result_does_not_touch_the_gpu(svc, fake_comfy):
    api = client_for(svc)
    ref = upload(api)
    design(api, ref)
    drain(svc, "generate", gen_handler(svc))
    r = design(api, ref)
    assert r.status_code == 200 and r.json()["state"] == "succeeded"
    assert httpx.get(f"{fake_comfy.url}/_fake/stats").json()["prompts"] == 1


def test_comfy_unreachable_retries_then_dead(tmp_path):
    svc = build(make_config(tmp_path, "http://127.0.0.1:9"))  # nothing listens on port 9
    api = client_for(svc)
    job_id = design(api, upload(api)).json()["id"]
    drain(svc, "generate", gen_handler(svc))
    job = api.get(f"/v1/designs/{job_id}").json()
    assert job["state"] == "dead" and job["attempts"] == job["max_attempts"]
    assert "ComfyUnavailable" in job["error"]


def test_rejected_graph_fails_without_retry(svc, fake_comfy):
    set_mode(fake_comfy, "reject")
    api = client_for(svc)
    job_id = design(api, upload(api)).json()["id"]
    drain(svc, "generate", gen_handler(svc))
    job = api.get(f"/v1/designs/{job_id}").json()
    assert job["state"] == "failed" and job["attempts"] == 1


def test_execution_error_is_retried(svc, fake_comfy):
    set_mode(fake_comfy, "error")
    api = client_for(svc)
    job_id = design(api, upload(api)).json()["id"]
    drain(svc, "generate", gen_handler(svc))
    job = api.get(f"/v1/designs/{job_id}").json()
    assert job["state"] == "dead" and "OutOfMemoryError" in job["error"]


def test_hung_comfy_times_out_and_retries(svc, fake_comfy):
    set_mode(fake_comfy, "hang")
    api = client_for(svc)
    job_id = design(api, upload(api)).json()["id"]
    run_one(svc, "generate", "w", gen_handler(svc))
    job = api.get(f"/v1/designs/{job_id}").json()
    assert job["state"] == "queued" and "no message from ComfyUI" in job["error"]


def test_worker_killed_mid_job_recovers(tmp_path, fake_comfy):
    """SIGKILL a real worker process while it holds a job; the next worker finishes it."""
    cfg = make_config(tmp_path, fake_comfy.url)
    cfg["comfyui"]["ws_idle_timeout_s"] = 60  # the killed worker must not give up on its own
    path = write_config(cfg, tmp_path)
    svc = build(cfg)
    api = client_for(svc)
    job_id = design(api, upload(api)).json()["id"]
    set_mode(fake_comfy, "hang")
    env = {
        **os.environ,
        "VOGUEFUSION_CONFIG": str(path),
        "PYTHONPATH": str(config.REPO_ROOT / "src"),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "voguefusion.worker", "generate"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 20
        while svc.store.get(job_id).substate != "sampling":
            assert time.monotonic() < deadline, svc.store.get(job_id)
            time.sleep(0.05)
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait()
    assert svc.store.get(job_id).state == "running"  # nobody knows it died yet
    set_mode(fake_comfy, "ok")
    time.sleep(cfg["queue"]["kinds"]["generate"]["lease_s"] + 0.2)
    drain(svc, "generate", gen_handler(svc))
    job = svc.store.get(job_id)
    assert job.state == "succeeded" and job.attempts == 2


def test_cost_job_end_to_end(svc):
    api = client_for(svc)
    sha = svc.blobs.put(two_tone_png())
    r = api.post("/v1/costs", json={"artifact": sha, "gsm": 100, "length_m": 5.5, "width_m": 1.1})
    assert r.status_code == 202
    drain(svc, "cost", cost_handler(svc))
    job = api.get(r.headers["location"]).json()
    assert job["state"] == "succeeded", job
    q = job["result"]
    assert [round(c["area_fraction"], 2) for c in q["colours"]] == [0.75, 0.25]
    assert q["total_cost"] == pytest.approx(sum(c["cost"] for c in q["colours"]))
    assert q["fabric_mass_kg"] == pytest.approx(100 * 5.5 * 1.1 / 1000)
    again = api.post(
        "/v1/costs", json={"artifact": sha, "gsm": 100, "length_m": 5.5, "width_m": 1.1}
    )
    assert again.status_code == 200 and again.json()["id"] == job["id"]  # cached quote


def test_cost_validation(svc):
    api = client_for(svc)
    assert api.post("/v1/costs", json={"artifact": "0" * 64}).status_code == 422
    sha = svc.blobs.put(two_tone_png())
    assert api.post("/v1/costs", json={"artifact": sha, "substrate": "silk"}).status_code == 422
    assert api.post("/v1/costs", json={"artifact": "../etc"}).status_code == 422


def test_cost_is_not_blocked_by_a_stuck_gpu_job(svc, fake_comfy):
    """Separate worker pools: a hung generation does not delay a quote."""
    cfg = svc.cfg
    cfg["comfyui"]["ws_idle_timeout_s"] = 30
    set_mode(fake_comfy, "hang")
    api = client_for(svc)
    gen_id = design(api, upload(api)).json()["id"]
    stop = threading.Event()
    t = threading.Thread(
        target=run_forever, args=(svc, "generate", make_handler(svc, "generate"), stop)
    )
    t.start()
    try:
        deadline = time.monotonic() + 10
        while svc.store.get(gen_id).substate != "sampling":
            assert time.monotonic() < deadline
            time.sleep(0.05)
        sha = svc.blobs.put(two_tone_png())
        cost_id = api.post("/v1/costs", json={"artifact": sha}).json()["id"]
        drain(svc, "cost", cost_handler(svc))
        assert svc.store.get(cost_id).state == "succeeded"
        assert svc.store.get(gen_id).state == "running"
    finally:
        stop.set()
        set_mode(fake_comfy, "ok")
        t.join(timeout=40)


def test_suggestions_empty_before_first_build_then_served(svc):
    api = client_for(svc)
    assert api.get("/v1/suggestions").json()["keywords"] == []
    from voguefusion.trends import build_store, write_store

    store = build_store(
        source="t",
        label="l",
        docs=[["paisley"]],
        seasonal=None,
        exclude=set(),
        top_n=3,
        recent_seasons=1,
        min_support=1,
        caveats=[],
    )
    write_store(svc.data_dir / svc.cfg["trends"]["store_dir"], store)
    assert api.get("/v1/suggestions").json()["keywords"] == ["paisley"]


def test_workflows_report_graph_defaults(svc):
    w = client_for(svc).get("/v1/workflows").json()
    assert w["workflows"]["v1"]["default_negative"] == "human"
    assert w["workflows"]["v1"]["default_batch"] == 1
    assert w["workflows"]["v2"]["default_batch"] == 3
    assert w["workflows"]["v2"]["reference_slots"] == 4


def test_costing_endpoint_gives_catalogue_and_fabric_defaults(svc):
    c = client_for(svc).get("/v1/costing").json()
    assert c["currency"] == "INR" and len(c["dyes"]) == 20
    assert {"name", "rgb", "price_per_kg"} <= set(c["dyes"][0])
    assert c["fabric"]["gsm_options"] == [70, 100, 120]
    assert c["default_substrate"] in c["substrates"]
    assert c["delta_e_tolerance"] == 2.0


def test_web_app_is_served_at_root_without_shadowing_the_api(svc):
    api = client_for(svc)
    r = api.get("/")
    assert r.status_code == 200 and "<title>VogueFusion</title>" in r.text
    assert api.get("/app.js").status_code == 200
    assert api.get("/healthz").json()["status"] == "ok"
    assert api.get("/v1/designs/nope").status_code == 404
