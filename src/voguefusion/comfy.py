"""Minimal client for the ComfyUI HTTP and websocket API.

Only the calls the GPU worker needs: upload a reference image, queue a
graph, follow its execution over the websocket, and download the outputs.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect


class ComfyUnavailable(Exception):
    """ComfyUI could not be reached or stopped responding. Retryable."""


class ComfyRejected(Exception):
    """ComfyUI refused the graph (validation error). Not retryable."""


class ComfyExecutionError(Exception):
    """A node raised during execution (e.g. out of memory). Retryable."""


@dataclass(frozen=True)
class OutputImage:
    filename: str
    subfolder: str
    type: str


class ComfyClient:
    def __init__(self, base_url: str, http_timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.ws_url = self.base_url.replace("http", "ws", 1) + "/ws"
        self.http = httpx.Client(base_url=self.base_url, timeout=http_timeout_s)

    def _call(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            return self.http.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise ComfyUnavailable(f"{method} {path}: {e!r}") from e

    def reachable(self) -> bool:
        try:
            return self.http.get("/system_stats").status_code == 200
        except httpx.HTTPError:
            return False

    def upload_image(self, name: str, data: bytes) -> None:
        r = self._call(
            "POST",
            "/upload/image",
            files={"image": (name, data, "image/png")},
            data={"overwrite": "true", "type": "input"},
        )
        if r.status_code != 200:
            raise ComfyUnavailable(f"upload {name}: HTTP {r.status_code}")

    def queue(self, graph: dict[str, Any], client_id: str) -> str:
        r = self._call("POST", "/prompt", json={"prompt": graph, "client_id": client_id})
        if r.status_code == 400:
            raise ComfyRejected(r.text[:2000])
        if r.status_code != 200:
            raise ComfyUnavailable(f"queue: HTTP {r.status_code}")
        return r.json()["prompt_id"]

    def outputs(self, prompt_id: str, node_id: str) -> list[OutputImage]:
        r = self._call("GET", f"/history/{prompt_id}")
        if r.status_code != 200:
            raise ComfyUnavailable(f"history: HTTP {r.status_code}")
        entry = r.json().get(prompt_id)
        if entry is None:
            raise ComfyUnavailable(f"history has no entry for {prompt_id}")
        images = entry.get("outputs", {}).get(node_id, {}).get("images", [])
        return [
            OutputImage(i["filename"], i.get("subfolder", ""), i.get("type", "output"))
            for i in images
        ]

    def download(self, image: OutputImage) -> bytes:
        r = self._call(
            "GET",
            "/view",
            params={"filename": image.filename, "subfolder": image.subfolder, "type": image.type},
        )
        if r.status_code != 200:
            raise ComfyUnavailable(f"view {image.filename}: HTTP {r.status_code}")
        return r.content

    def run(
        self,
        graph: dict[str, Any],
        on_event: Callable[[str | None], None],
        heartbeat_s: float,
        idle_timeout_s: float,
        job_timeout_s: float,
    ) -> str:
        """Queue graph and block until it finishes. Returns the prompt id.

        on_event is called with the executing node id on every node change
        and with None on every heartbeat, so the caller can renew its lease.
        """
        client_id = uuid.uuid4().hex
        try:
            ws = connect(f"{self.ws_url}?clientId={client_id}", open_timeout=heartbeat_s)
        except (OSError, WebSocketException, TimeoutError) as e:
            raise ComfyUnavailable(f"websocket: {e!r}") from e
        with ws:
            prompt_id = self.queue(graph, client_id)
            started = last_message = time.monotonic()
            while True:
                now = time.monotonic()
                if now - started > job_timeout_s:
                    raise ComfyUnavailable(f"job exceeded {job_timeout_s}s")
                if now - last_message > idle_timeout_s:
                    raise ComfyUnavailable(f"no message from ComfyUI for {idle_timeout_s}s")
                try:
                    raw = ws.recv(timeout=heartbeat_s)
                except TimeoutError:
                    on_event(None)
                    continue
                except WebSocketException as e:
                    raise ComfyUnavailable(f"websocket closed: {e!r}") from e
                last_message = time.monotonic()
                if isinstance(raw, bytes):  # preview frames
                    continue
                msg = json.loads(raw)
                data = msg.get("data", {})
                if data.get("prompt_id") not in (None, prompt_id):
                    continue
                kind = msg.get("type")
                if kind == "execution_error":
                    raise ComfyExecutionError(
                        f"{data.get('node_type')}: {data.get('exception_type')}: "
                        f"{data.get('exception_message', '')[:500]}"
                    )
                if kind == "executing":
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        return prompt_id
                    on_event(data.get("node"))
                elif kind == "execution_success" and data.get("prompt_id") == prompt_id:
                    return prompt_id
                else:
                    on_event(None)
