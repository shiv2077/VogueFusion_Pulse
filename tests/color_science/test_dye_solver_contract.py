"""Contract tests for the dye-matching solver (ADR-004, ADR-005, ADR-007, ADR-008).

This file was written as an xfail(strict=True) specification against a stub
that reproduced the incumbent's choices (nearest dye by raw sRGB distance,
single dye, no cost or tolerance reasoning). The stub is gone; every test now
runs the real solver in src/costing/solver.py.

Three assertions changed shape when the stub was replaced, because they were
written around the stub's fixed concentration of 1.0:

- The substitution and cheapest-within-tolerance tests compared against the
  cost of a single dye at unit concentration. A solver that minimises cost
  subject to a tolerance moves to the tolerance boundary, so its cost is at
  most that figure, not equal to it. They now assert "at most", plus that the
  expensive near-duplicate dye is never used.
- The seed tests were written for a genetic algorithm. The solver is now
  exhaustive and deterministic (ADR-008), so they assert repeat and
  catalogue-order invariance instead of a variance bound across seeds.
"""

import inspect

import numpy as np
import pytest

from costing.color import perceptual as p
from costing.solver import Dye, SolverParams, solve

CATALOGUE = {
    "reactive_blue_19": {"reflectance": np.array([0.10, 0.20, 0.80]), "cost": 42.0},
    "reactive_yellow_15": {"reflectance": np.array([0.85, 0.75, 0.08]), "cost": 28.0},
    "reactive_red_195": {"reflectance": np.array([0.80, 0.10, 0.45]), "cost": 51.0},
    "reactive_black_5": {"reflectance": np.array([0.04, 0.04, 0.04]), "cost": 19.0},
    # dE2000 = 0.77 from reactive_blue_19 -- invisible -- at 2.3x the price.
    # This is the substitution opportunity the incumbent could not see. Do
    # not change this value: it is tuned so the substitution is real.
    "direct_blue_86": {"reflectance": np.array([0.105, 0.21, 0.785]), "cost": 96.0},
}

PARAMS = SolverParams(
    max_dyes=3,
    max_concentration=3.0,
    min_concentration=0.01,
    polish_top_k=3,
    polish_maxiter=200,
    polish_ftol=1e-9,
    polish_margin=1e-4,
)
TOLERANCE = 2.0
DEPTH_G_PER_KG = 20.0


def _dyes(catalogue=CATALOGUE):
    return [Dye(n, tuple(d["reflectance"]), d["cost"]) for n, d in catalogue.items()]


def _solver(target_rgb, catalogue=CATALOGUE, tolerance=TOLERANCE, substrate=(1.0, 1.0, 1.0)):
    return solve(
        target_rgb,
        _dyes(catalogue),
        params=PARAMS,
        tolerance=tolerance,
        reference_depth_g_per_kg=DEPTH_G_PER_KG,
        substrate=substrate,
    ).as_dict()


def _names(result):
    return [d["name"] for d in result["dyes"]]


def test_solution_meets_delta_e_tolerance():
    result = _solver(CATALOGUE["reactive_blue_19"]["reflectance"])
    assert result["feasible"] is True
    assert result["delta_e"] <= TOLERANCE


def test_solver_can_reach_two_dye_target():
    target = p.mix_subtractive(
        np.array(
            [
                CATALOGUE["reactive_blue_19"]["reflectance"],
                CATALOGUE["reactive_yellow_15"]["reflectance"],
            ]
        ),
        np.array([0.6, 0.4]),
    )
    result = _solver(target)
    assert result["feasible"] is True
    assert len(result["dyes"]) >= 2
    assert {"reactive_blue_19", "reactive_yellow_15"} <= set(_names(result))


def test_solver_substitutes_perceptually_identical_cheaper_dye():
    # The swatch matches direct_blue_86 exactly. reactive_blue_19 is ΔE 0.77
    # away at less than half the price; the solver must never pay for 86.
    result = _solver(CATALOGUE["direct_blue_86"]["reflectance"])
    assert "direct_blue_86" not in _names(result)
    assert _names(result)[0] == "reactive_blue_19"
    assert result["cost"] <= CATALOGUE["reactive_blue_19"]["cost"]


def test_returned_formulation_is_cheapest_within_tolerance():
    target = CATALOGUE["direct_blue_86"]["reflectance"]
    feasible_singles = [
        name
        for name, dye in CATALOGUE.items()
        if p.delta_e_2000(p.srgb_to_lab(target), p.srgb_to_lab(dye["reflectance"])) <= TOLERANCE
    ]
    cheapest_single = min(CATALOGUE[n]["cost"] for n in feasible_singles)
    result = _solver(target)
    assert result["feasible"] is True
    assert result["cost"] <= cheapest_single


def test_cost_matches_brute_force_grid_over_one_and_two_dye_recipes():
    """No 1- or 2-dye recipe on a fine grid is cheaper than the solver's answer."""
    target = CATALOGUE["direct_blue_86"]["reflectance"]
    dyes = _dyes()
    ks = p.reflectance_to_ks(np.array([d.reflectance for d in dyes]))
    costs = np.array([d.unit_cost for d in dyes])
    target_lab = p.srgb_to_lab(target)
    grid = np.linspace(0.0, 1.5, 151)
    best = np.inf
    for i in range(len(dyes)):
        for j in range(i, len(dyes)):
            a, b = np.meshgrid(grid, grid)
            conc = np.zeros((a.size, len(dyes)))
            conc[:, i] += a.ravel()
            conc[:, j] += b.ravel()
            refl = np.clip(p.ks_to_reflectance(conc @ ks), 0.0, 1.0)
            ok = p.delta_e_2000(p.srgb_to_lab(refl), target_lab) <= TOLERANCE
            if ok.any():
                best = min(best, float((conc[ok] @ costs).min()))
    assert _solver(target)["cost"] <= best * (1 + 1e-3)


def test_unreachable_targets_reported_infeasible():
    target = np.array([0.0, 1.0, 1.0])  # outside the catalogue's achievable gamut
    result = _solver(target)
    assert result["feasible"] is False
    assert result["delta_e"] > TOLERANCE


def test_convergence_status_returned():
    result = _solver(CATALOGUE["reactive_blue_19"]["reflectance"])
    assert result["convergence_status"] in {"converged", "max_iterations"}


def test_concentrations_reported_in_g_per_kg():
    result = _solver(CATALOGUE["reactive_blue_19"]["reflectance"])
    g = result["concentration_g_per_kg"]
    assert g and all(v > 0 for v in g.values())
    for dose in result["dyes"]:
        assert dose["concentration_g_per_kg"] == pytest.approx(
            dose["concentration"] * DEPTH_G_PER_KG
        )


def test_substrate_is_solver_input():
    assert "substrate" in inspect.signature(solve).parameters
    target = CATALOGUE["reactive_blue_19"]["reflectance"] * 0.9 + 0.05
    white = _solver(target)
    cotton = _solver(target, substrate=(0.92, 0.89, 0.82))
    assert white["dyes"] != cotton["dyes"]


def test_deterministic_on_repeat():
    target = CATALOGUE["reactive_blue_19"]["reflectance"]
    assert _solver(target) == _solver(target)


def test_cost_independent_of_catalogue_order():
    target = CATALOGUE["direct_blue_86"]["reflectance"]
    reversed_catalogue = dict(reversed(list(CATALOGUE.items())))
    a, b = _solver(target), _solver(target, catalogue=reversed_catalogue)
    assert a["cost"] == pytest.approx(b["cost"], rel=1e-6)
    assert sorted(_names(a)) == sorted(_names(b))
