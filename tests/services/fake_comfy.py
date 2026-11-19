"""A stand-in for ComfyUI's HTTP + websocket API, for tests without a GPU.

Behaviour is switched with POST /_fake/mode {"mode": ..., "delay_s": ...}:
  ok      run every node, emit executing messages, then produce one image per batch item
  reject  answer POST /prompt with 400, as ComfyUI does for a graph that fails validation
  error   emit execution_error at the KSampler
  hang    reach the KSampler and then go silent
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from PIL import Image


def make_app() -> FastAPI:
    app = FastAPI()
    st: dict[str, Any] = {
        "mode": "ok",
        "delay_s": 0.01,
        "uploads": {},
        "history": {},
        "queues": {},
        "prompts": [],
    }

    def queue_for(cid: str) -> asyncio.Queue:
        return st["queues"].setdefault(cid, asyncio.Queue())

    @app.get("/system_stats")
    def system_stats() -> dict:
        return {"system": {"fake": True}}

    @app.post("/_fake/mode")
    async def set_mode(req: Request) -> dict:
        st.update(await req.json())
        return {"mode": st["mode"]}

    @app.get("/_fake/stats")
    def stats() -> dict:
        return {"prompts": len(st["prompts"]), "uploads": sorted(st["uploads"])}

    @app.post("/upload/image")
    async def upload(image: UploadFile) -> dict:
        st["uploads"][image.filename] = await image.read()
        return {"name": image.filename, "subfolder": "", "type": "input"}

    async def execute(prompt_id: str, cid: str, graph: dict) -> None:
        q = queue_for(cid)
        ks = next(k for k, n in graph.items() if n["class_type"] == "KSampler")
        save = next(k for k, n in graph.items() if n["class_type"] == "SaveImage")
        for node in sorted(graph, key=int):
            await asyncio.sleep(st["delay_s"])
            await q.put({"type": "executing", "data": {"node": node, "prompt_id": prompt_id}})
            if node == ks and st["mode"] == "hang":
                return
            if node == ks and st["mode"] == "error":
                await q.put(
                    {
                        "type": "execution_error",
                        "data": {
                            "prompt_id": prompt_id,
                            "node_type": "KSampler",
                            "exception_type": "torch.OutOfMemoryError",
                            "exception_message": "fake OOM",
                        },
                    }
                )
                return
        seed = graph[ks]["inputs"]["seed"]
        batch = next(
            n["inputs"]["amount"] for n in graph.values() if n["class_type"] == "RepeatLatentBatch"
        )
        st["history"][prompt_id] = {
            "outputs": {
                save: {
                    "images": [
                        {"filename": f"{seed}_{i}.png", "subfolder": "", "type": "output"}
                        for i in range(batch)
                    ]
                }
            }
        }
        await q.put({"type": "executing", "data": {"node": None, "prompt_id": prompt_id}})

    @app.post("/prompt")
    async def prompt(req: Request) -> Response:
        body = await req.json()
        if st["mode"] == "reject":
            return JSONResponse({"error": {"type": "prompt_outputs_failed_validation"}}, 400)
        for n in body["prompt"].values():
            if n["class_type"] == "LoadImage" and n["inputs"]["image"] not in st["uploads"]:
                return JSONResponse({"error": "missing input image"}, 400)
        pid = uuid.uuid4().hex
        st["prompts"].append(body["prompt"])
        asyncio.get_running_loop().create_task(execute(pid, body["client_id"], body["prompt"]))
        return JSONResponse({"prompt_id": pid, "number": len(st["prompts"])})

    @app.get("/history/{prompt_id}")
    def history(prompt_id: str) -> dict:
        h = st["history"].get(prompt_id)
        return {prompt_id: h} if h else {}

    @app.get("/view")
    def view(filename: str, subfolder: str = "", type: str = "output") -> Response:
        seed = int(filename.split("_")[0])
        index = int(filename.split("_")[1].split(".")[0])
        colour = ((seed * 37 + index) % 256, (seed * 91) % 256, (seed * 13 + 7 * index) % 256)
        buf = io.BytesIO()
        Image.new("RGB", (96, 128), colour).save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        cid = socket.query_params["clientId"]
        q = queue_for(cid)
        await socket.send_text(json.dumps({"type": "status", "data": {"sid": cid}}))
        try:
            while True:
                await socket.send_text(json.dumps(await q.get()))
        except WebSocketDisconnect:
            return

    return app


class FakeComfy:
    """Run the fake on 127.0.0.1:<free port> in a background thread."""

    def __init__(self) -> None:
        self.app = make_app()
        self.server = uvicorn.Server(
            uvicorn.Config(
                self.app,
                host="127.0.0.1",
                port=0,
                log_level="warning",
                lifespan="off",
                timeout_graceful_shutdown=0.2,
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> FakeComfy:
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("fake ComfyUI did not start")
            time.sleep(0.01)
        port = self.server.servers[0].sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)
