"""End-to-end check through the public API only: upload, generate, cost.

Needs the stack running (scripts/vf.sh start). For each workflow named on the
command line it uploads the four original references, generates with a
fresh random seed, waits, then costs the first image. Prints one JSON line
per step and exits non-zero on any failure.

Run: PYTHONPATH=src python scripts/e2e.py v1 v2
"""

from __future__ import annotations

import json
import sys
import time
import uuid

import httpx

from voguefusion import config

REFS = ["image_1.jpg", "image_3.jpg", "image_13.jpg", "image_26.jpg"]
TERMINAL = {"succeeded", "failed", "dead"}


def wait(h: httpx.Client, location: str, timeout_s: float) -> tuple[dict, list[str]]:
    substates: list[str] = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        j = h.get(location).json()
        if j.get("substate") and (not substates or substates[-1] != j["substate"]):
            substates.append(j["substate"])
        if j["state"] in TERMINAL:
            return j, substates
        time.sleep(0.25)
    raise TimeoutError(location)


def main() -> int:
    cfg = config.load()
    workflows = sys.argv[1:] or [cfg["generate"]["default_workflow"]]
    base = f"http://{cfg['api']['host']}:{cfg['api']['port']}"
    ref_dir = config.REPO_ROOT / "dataset/initial_dataset/Saree"
    ok = True
    with httpx.Client(base_url=base, timeout=30) as h:
        refs = [
            h.post("/v1/references", files={"file": (n, (ref_dir / n).read_bytes())}).json()[
                "sha256"
            ]
            for n in REFS
        ]
        for wf in workflows:
            t0 = time.monotonic()
            r = h.post(
                "/v1/designs",
                json={
                    "workflow": wf,
                    "prompt": "traditional, floral, saree border",
                    "references": refs,
                },
                headers={"Idempotency-Key": uuid.uuid4().hex},
            )
            if r.status_code not in (200, 202):
                print(
                    json.dumps(
                        {"step": "submit", "workflow": wf, "status": r.status_code, "body": r.text}
                    )
                )
                ok = False
                continue
            job, subs = wait(h, r.headers["location"], cfg["comfyui"]["job_timeout_s"] * 3)
            row = {
                "step": "generate",
                "workflow": wf,
                "state": job["state"],
                "attempts": job["attempts"],
                "end_to_end_s": round(time.monotonic() - t0, 2),
                "substates": subs,
                "error": job.get("error"),
            }
            if job["state"] == "succeeded":
                res = job["result"]
                row.update(
                    seed=res["seed"],
                    images=len(res["artifacts"]),
                    comfy_s=res["comfy_seconds"],
                    models=res["models"],
                )
            print(json.dumps(row))
            if job["state"] != "succeeded":
                ok = False
                continue
            art = job["result"]["artifacts"][0]
            img = h.get(f"/v1/artifacts/{art}")
            ok &= img.status_code == 200 and img.content[:8] == b"\x89PNG\r\n\x1a\n"
            t1 = time.monotonic()
            c = h.post("/v1/costs", json={"artifact": art})
            cj, _ = wait(h, c.headers["location"], 120)
            q = cj.get("result") or {}
            print(
                json.dumps(
                    {
                        "step": "cost",
                        "workflow": wf,
                        "state": cj["state"],
                        "seconds": round(time.monotonic() - t1, 2),
                        "total_cost": q.get("total_cost"),
                        "currency": q.get("currency"),
                        "all_feasible": q.get("all_feasible"),
                        "colours": [
                            (
                                c["hex"],
                                round(c["area_fraction"], 3),
                                round(c["delta_e"], 2),
                                [d["name"] for d in c["dyes"]],
                            )
                            for c in q.get("colours", [])
                        ],
                    }
                )
            )
            ok &= cj["state"] == "succeeded"
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
