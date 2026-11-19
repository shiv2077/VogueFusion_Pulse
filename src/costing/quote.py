"""Dye cost quote for a printed design.

Each dominant colour covers a share of the fabric area and gets its own
formulation. Dye mass per colour is

    fabric_mass_kg * area_fraction * concentration_g_per_kg / 1000

and its cost is that mass times the dye's price per kg. Equivalently, a
colour's cost is fabric_mass_kg * area_fraction * formulation.cost, since
formulation.cost is already per kg of fabric.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from costing.catalogue import CatalogueEntry
from costing.extract import Colour
from costing.solver import Dye, Formulation, SolverParams, solve


@dataclass(frozen=True)
class Fabric:
    gsm: float
    length_m: float
    width_m: float
    substrate_srgb: tuple[float, float, float]


@dataclass(frozen=True)
class DyeLine:
    name: str
    concentration_g_per_kg: float
    dye_kg: float
    price_per_kg: float
    cost: float


@dataclass(frozen=True)
class ColourLine:
    hex: str
    srgb: tuple[float, float, float]
    area_fraction: float
    feasible: bool
    delta_e: float
    convergence_status: str
    predicted_hex: str
    dyes: tuple[DyeLine, ...]
    cost: float


@dataclass(frozen=True)
class Quote:
    currency: str
    fabric_mass_kg: float
    colours: tuple[ColourLine, ...]
    total_cost: float
    all_feasible: bool
    solver_version: str

    def as_dict(self) -> dict:
        return asdict(self)


def fabric_mass_kg(f: Fabric) -> float:
    """Grams per square metre times area, in kg."""
    return f.gsm * f.length_m * f.width_m / 1000.0


def colour_line(
    colour: Colour,
    formulation: Formulation,
    prices: dict[str, float],
    mass_kg: float,
) -> ColourLine:
    """Cost one colour from its own formulation; dye names and prices travel together."""
    covered = mass_kg * colour.fraction
    dyes = tuple(
        DyeLine(
            name=d.name,
            concentration_g_per_kg=d.concentration_g_per_kg,
            dye_kg=covered * d.concentration_g_per_kg / 1000.0,
            price_per_kg=prices[d.name],
            cost=covered * d.concentration_g_per_kg / 1000.0 * prices[d.name],
        )
        for d in formulation.dyes
    )
    return ColourLine(
        hex=colour.hex,
        srgb=colour.srgb,
        area_fraction=colour.fraction,
        feasible=formulation.feasible,
        delta_e=formulation.delta_e,
        convergence_status=formulation.convergence_status,
        predicted_hex=Colour(formulation.predicted_srgb, 0.0).hex,
        dyes=dyes,
        cost=sum(d.cost for d in dyes),
    )


def quote(
    colours: list[Colour],
    entries: list[CatalogueEntry],
    dyes: list[Dye],
    fabric: Fabric,
    *,
    params: SolverParams,
    tolerance: float,
    reference_depth_g_per_kg: float,
    currency: str,
    solver_version: str,
) -> Quote:
    """Solve each colour independently and total the dye cost for the fabric piece."""
    prices = {e.name: e.price_per_kg for e in entries}
    mass = fabric_mass_kg(fabric)
    lines = tuple(
        colour_line(
            c,
            solve(
                c.srgb,
                dyes,
                params=params,
                tolerance=tolerance,
                reference_depth_g_per_kg=reference_depth_g_per_kg,
                substrate=fabric.substrate_srgb,
            ),
            prices,
            mass,
        )
        for c in colours
    )
    return Quote(
        currency=currency,
        fabric_mass_kg=mass,
        colours=lines,
        total_cost=sum(line.cost for line in lines),
        all_feasible=all(line.feasible for line in lines),
        solver_version=solver_version,
    )
