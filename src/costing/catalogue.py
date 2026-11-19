"""Load the dye catalogue and convert supplier prices into solver unit costs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from costing.solver import Dye


@dataclass(frozen=True)
class CatalogueEntry:
    name: str
    srgb_255: tuple[int, int, int]
    price_per_kg: float  # currency per kg of dye powder, from the supplier price list


def load_entries(path: Path) -> list[CatalogueEntry]:
    """Read dye_detail.json: [{"color": name, "rgb": [r, g, b], "price": per kg}, ...]."""
    rows = json.loads(Path(path).read_text())
    return [CatalogueEntry(r["color"], tuple(r["rgb"]), float(r["price"])) for r in rows]


def unit_cost(price_per_kg: float, reference_depth_g_per_kg: float) -> float:
    """Cost of unit concentration per kg of fabric: price/kg dye * kg dye per kg fabric."""
    return price_per_kg * reference_depth_g_per_kg / 1000.0


def to_dyes(
    entries: list[CatalogueEntry],
    reference_depth_g_per_kg: float,
    reflectance_floor: float,
    reflectance_ceiling: float,
) -> list[Dye]:
    """Solver dyes. Swatch channels are clamped to [floor, ceiling].

    A swatch channel of exactly 0 means K/S of about 5e5, which lets a
    millionth of a unit of that dye produce any colour. No real dyed fabric
    reflects 0% or 100%, so the clamp keeps the placeholder catalogue colours
    physically plausible.
    """
    return [
        Dye(
            name=e.name,
            reflectance=tuple(
                float(x)
                for x in np.clip(
                    np.array(e.srgb_255) / 255.0, reflectance_floor, reflectance_ceiling
                )
            ),
            unit_cost=unit_cost(e.price_per_kg, reference_depth_g_per_kg),
        )
        for e in entries
    ]
