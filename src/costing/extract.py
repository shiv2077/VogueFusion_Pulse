"""Dominant colours of a design image, clustered in CIE Lab.

Clustering runs in Lab rather than sRGB so that cluster boundaries follow
perceived colour difference (ADR-004).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from costing.color import perceptual as p


@dataclass(frozen=True)
class Colour:
    srgb: tuple[float, float, float]  # in [0, 1]
    fraction: float  # share of image area

    @property
    def hex(self) -> str:
        return "#" + "".join(f"{round(c * 255):02x}" for c in self.srgb)


def load_pixels(image_bytes: bytes, max_side_px: int) -> np.ndarray:
    """Decode to RGB, shrink so the longer side is <= max_side_px; (N, 3) in [0, 1]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((max_side_px, max_side_px))
    return np.asarray(img, dtype=np.float64).reshape(-1, 3) / 255.0


def dominant_colours(pixels: np.ndarray, n_colours: int, n_init: int, seed: int) -> list[Colour]:
    """K-means in Lab. Returns colours sorted by area, largest first. Deterministic for a seed."""
    lab = p.srgb_to_lab(pixels)
    k = min(n_colours, len(np.unique(pixels, axis=0)))
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(lab)
    counts = np.bincount(km.labels_, minlength=k)
    srgb = np.clip(p.lab_to_srgb(km.cluster_centers_), 0.0, 1.0)
    order = np.argsort(-counts, kind="stable")
    total = counts.sum()
    return [Colour(tuple(float(x) for x in srgb[i]), float(counts[i] / total)) for i in order]
