import copy
import io
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from tests.services.fake_comfy import FakeComfy
from voguefusion import config
from voguefusion.services import build


def make_config(tmp_path: Path, comfy_url: str = "http://127.0.0.1:9") -> dict:
    """The real config with fast timings, a temp data dir and the given ComfyUI url."""
    cfg = copy.deepcopy(config.load(str(config.DEFAULT_CONFIG)))
    cfg["data_dir_default"] = str(tmp_path / "data")
    cfg["data_dir_env"] = "VOGUEFUSION_TEST_DATA_UNSET"
    cfg["comfyui"]["url"] = comfy_url
    cfg["comfyui"].update(http_timeout_s=2, ws_idle_timeout_s=1.5, job_timeout_s=10)
    cfg["queue"].update(poll_interval_s=0.02, heartbeat_s=0.2)
    for k in cfg["queue"]["kinds"].values():
        k.update(lease_s=2.0, retry_backoff_s=0.05, max_depth=4)
    return cfg


def write_config(cfg: dict, tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def png_bytes(colour=(200, 40, 40), size=(80, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def two_tone_png() -> bytes:
    a = np.zeros((100, 80, 3), dtype=np.uint8)
    a[:, :60] = (200, 40, 60)
    a[:, 60:] = (30, 60, 170)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def fake_comfy():
    with FakeComfy() as fake:
        yield fake


@pytest.fixture
def svc(tmp_path, fake_comfy):
    return build(make_config(tmp_path, fake_comfy.url))
