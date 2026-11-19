"""Measure the running stack on this machine. Results go to $VOGUEFUSION_DATA/measurements/.

  profile   per-job timings: cold start (restarts ComfyUI), warm v1, first v2 (LoRA
            load), warm v2, batch 3, per-substate durations, peak VRAM and ComfyUI
            RSS, and costing time.
  load      open-loop Poisson arrivals of v1 batch-1 jobs at several utilisations;
            records queue wait, end-to-end latency, 429s, and compares the measured
            mean queue wait with the M/G/1 Pollaczek-Khinchine prediction.

Every job uses a fresh seed, so ComfyUI's own result cache never short-cuts a run.

Run: PYTHONPATH=src python scripts/measure.py profile
     PYTHONPATH=src python scripts/measure.py load --rho 0.5 0.8 0.95 1.3 --jobs 20
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from voguefusion import config
from voguefusion.jobstore import percentile

REFS = ["image_1.jpg", "image_3.jpg", "image_13.jpg", "image_26.jpg"]
TERMINAL = {"succeeded", "failed", "dead"}


class Sampler:
    """Samples GPU memory and ComfyUI resident memory once per interval in a thread."""

    def __init__(self, comfy_pid_file: Path, interval_s: float = 0.5) -> None:
        self.pid_file = comfy_pid_file
        self.interval = interval_s
        self.vram: list[int] = []
        self.rss: list[int] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self) -> None:
        while not self.stop.wait(self.interval):
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            if out:
                self.vram.append(int(out.splitlines()[0]))
            try:
                pid = self.pid_file.read_text().strip()
                children = subprocess.run(
                    ["pgrep", "-P", pid], capture_output=True, text=True
                ).stdout.split()
                total = 0
                for p in [pid, *children]:
                    for line in Path(f"/proc/{p}/status").read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            total += int(line.split()[1])
                self.rss.append(total // 1024)
            except (OSError, ValueError):
                pass

    def __enter__(self) -> Sampler:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop.set()
        self.thread.join()


def api(cfg: dict) -> httpx.Client:
    return httpx.Client(base_url=f"http://{cfg['api']['host']}:{cfg['api']['port']}", timeout=30)


def upload_refs(h: httpx.Client) -> list[str]:
    d = config.REPO_ROOT / "dataset/initial_dataset/Saree"
    return [
        h.post("/v1/references", files={"file": (n, (d / n).read_bytes())}).json()["sha256"]
        for n in REFS
    ]


def follow(h: httpx.Client, location: str, poll_s: float = 0.1) -> tuple[dict, dict[str, float]]:
    """Poll until terminal; return the job and seconds spent in each substate."""
    seen: list[tuple[float, str]] = []
    while True:
        j = h.get(location).json()
        now = time.monotonic()
        label = j.get("substate") or j["state"]
        if not seen or seen[-1][1] != label:
            seen.append((now, label))
        if j["state"] in TERMINAL:
            break
        time.sleep(poll_s)
    durations: dict[str, float] = {}
    for (t0, lab), (t1, _) in zip(seen, seen[1:], strict=False):
        durations[lab] = round(durations.get(lab, 0.0) + t1 - t0, 2)
    return j, durations


def one_job(h: httpx.Client, refs: list[str], workflow: str, batch: int) -> dict:
    body = {
        "workflow": workflow,
        "prompt": "traditional, floral, saree border",
        "references": refs,
        "batch": batch,
        "seed": random.getrandbits(48),
    }
    t0 = time.monotonic()
    r = h.post("/v1/designs", json=body)
    j, subs = follow(h, r.headers["location"])
    return {
        "workflow": workflow,
        "batch": batch,
        "state": j["state"],
        "client_e2e_s": round(time.monotonic() - t0, 2),
        "service_s": round(j["finished_at"] - j["started_at"], 2),
        "comfy_s": (j.get("result") or {}).get("comfy_seconds"),
        "substates_s": subs,
        "artifacts": (j.get("result") or {}).get("artifacts", []),
    }


def restart_comfy(cfg: dict) -> float:
    vf = str(config.REPO_ROOT / "scripts/vf.sh")
    subprocess.run([vf, "stop", "comfyui"], check=True, capture_output=True)
    t0 = time.monotonic()
    subprocess.run([vf, "start", "comfyui"], check=True, capture_output=True)
    url = cfg["comfyui"]["url"] + "/system_stats"
    while True:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return round(time.monotonic() - t0, 2)
        except httpx.HTTPError:
            pass
        time.sleep(0.5)


def profile(cfg: dict, out: Path) -> dict:
    data = config.data_dir(cfg)
    results: dict = {"steps_per_image": 10}
    with api(cfg) as h:
        refs = upload_refs(h)
        results["comfy_process_start_s"] = restart_comfy(cfg)
        with Sampler(data / "run/comfyui.pid") as s:
            runs = [("cold_v1", "v1", 1)] + [(f"warm_v1_{i}", "v1", 1) for i in range(3)]
            runs += [("first_v2_lora_load", "v2", 1)] + [
                (f"warm_v2_{i}", "v2", 1) for i in range(2)
            ]
            runs += [
                ("v1_after_v2", "v1", 1),
                ("warm_v1_batch3", "v1", 3),
                ("warm_v2_batch3", "v2", 3),
            ]
            for name, wf, batch in runs:
                results[name] = one_job(h, refs, wf, batch)
                print(name, json.dumps(results[name]))
        results["peak_vram_mib"] = max(s.vram, default=None)
        results["peak_comfy_rss_mib"] = max(s.rss, default=None)
        results["vram_total_mib"] = int(
            subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        warm = [results[f"warm_v1_{i}"]["substates_s"].get("sampling", 0) for i in range(3)]
        results["warm_v1_sampling_s_per_step"] = round(statistics.mean(warm) / 10, 3)
        results["metrics_after"] = h.get("/v1/metrics").json()
        cost_times = []
        for name in ("warm_v1_0", "warm_v1_1", "warm_v1_2"):
            t0 = time.monotonic()
            c = h.post("/v1/costs", json={"artifact": results[name]["artifacts"][0]})
            cj, subs = follow(h, c.headers["location"])
            cost_times.append(
                {
                    "state": cj["state"],
                    "client_s": round(time.monotonic() - t0, 2),
                    "service_s": round(cj["finished_at"] - cj["started_at"], 2),
                    "substates_s": subs,
                }
            )
        results["costing"] = cost_times
    return results


def load(cfg: dict, rhos: list[float], jobs: int, service_s: float, cv2: float) -> dict:
    """Open-loop Poisson load at utilisation rho = lambda * E[S] for one GPU worker."""
    rng = random.Random(0)
    report: dict = {"service_s_assumed_for_rates": service_s, "service_cv2": cv2, "runs": []}
    with api(cfg) as h:
        refs = upload_refs(h)
        for rho in rhos:
            lam = rho / service_s
            lock = threading.Lock()
            rows: list[dict] = []
            threads = []

            def submit(rows: list[dict] = rows, lock: threading.Lock = lock) -> None:
                with api(cfg) as hc:
                    t0 = time.monotonic()
                    r = hc.post(
                        "/v1/designs",
                        json={
                            "workflow": "v1",
                            "prompt": "load test",
                            "references": refs,
                            "seed": random.getrandbits(48),
                        },
                    )
                    if r.status_code == 429:
                        with lock:
                            rows.append(
                                {"status": 429, "retry_after": int(r.headers["retry-after"])}
                            )
                        return
                    j, _ = follow(hc, r.headers["location"], poll_s=0.25)
                    with lock:
                        rows.append(
                            {
                                "status": r.status_code,
                                "state": j["state"],
                                "queue_wait_s": j["started_at"] - j["created_at"],
                                "service_s": j["finished_at"] - j["started_at"],
                                "e2e_s": j["finished_at"] - j["created_at"],
                                "client_s": time.monotonic() - t0,
                            }
                        )

            started = time.monotonic()
            for _ in range(jobs):
                t = threading.Thread(target=submit)
                t.start()
                threads.append(t)
                time.sleep(rng.expovariate(lam))
            for t in threads:
                t.join()
            ok = [r for r in rows if r.get("state") == "succeeded"]
            waits = [r["queue_wait_s"] for r in ok]
            services = [r["service_s"] for r in ok]
            e2e = [r["e2e_s"] for r in ok]
            es = statistics.mean(services) if services else service_s
            es2 = statistics.mean(s * s for s in services) if services else service_s**2
            measured_rho = lam * es
            pk = lam * es2 / (2 * (1 - measured_rho)) if measured_rho < 1 else None
            run = {
                "target_rho": rho,
                "lambda_per_s": round(lam, 4),
                "jobs": jobs,
                "duration_s": round(time.monotonic() - started, 1),
                "rejected_429": sum(1 for r in rows if r.get("status") == 429),
                "succeeded": len(ok),
                "other": [
                    r for r in rows if r.get("status") != 429 and r.get("state") != "succeeded"
                ],
                "service_mean_s": round(es, 2),
                "service_cv2": round(statistics.pvariance(services) / es**2, 4)
                if len(services) > 1
                else None,
                "measured_rho": round(measured_rho, 3),
                "queue_wait_mean_s": round(statistics.mean(waits), 2) if waits else None,
                "queue_wait_p50_s": round(percentile(waits, 50), 2) if waits else None,
                "queue_wait_p95_s": round(percentile(waits, 95), 2) if waits else None,
                "e2e_p50_s": round(percentile(e2e, 50), 2) if e2e else None,
                "e2e_p95_s": round(percentile(e2e, 95), 2) if e2e else None,
                "mg1_predicted_wait_mean_s": None if pk is None else round(pk, 2),
            }
            print(json.dumps(run))
            report["runs"].append(run)
            while h.get("/v1/metrics").json()["generate"]["depth"]:
                time.sleep(1)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("profile")
    lp = sub.add_parser("load")
    lp.add_argument("--rho", type=float, nargs="+", default=[0.5, 0.8, 0.95, 1.3])
    lp.add_argument("--jobs", type=int, default=20)
    lp.add_argument(
        "--service-s", type=float, required=True, help="mean warm service time from profile"
    )
    lp.add_argument("--service-cv2", type=float, default=0.0)
    args = ap.parse_args()
    cfg = config.load()
    out = config.data_dir(cfg) / "measurements"
    out.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    meta = {
        "utc": stamp,
        "gpu": gpu,
        "config_max_depth": cfg["queue"]["kinds"]["generate"]["max_depth"],
    }
    if args.cmd == "profile":
        res = profile(cfg, out)
    else:
        res = load(cfg, args.rho, args.jobs, args.service_s, args.service_cv2)
    path = out / f"{args.cmd}-{stamp}.json"
    path.write_text(json.dumps({"meta": meta, **res}, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
