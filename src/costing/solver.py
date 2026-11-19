"""Dye formulation solver: minimise cost subject to CIEDE2000 <= tolerance.

Mixtures are predicted with single-constant Kubelka-Munk (ADR-005) and
compared to the target with CIEDE2000 (ADR-004). Accuracy is a constraint,
not a weighted term (ADR-007).

Search (ADR-008). Every dye set of size 1..max_dyes is enumerated; there are
1,350 for the 20-dye catalogue and 3 dyes per recipe. K/S is linear in
concentration, so each set gets a closed-form least-squares concentration in
K/S space in one batched solve. The cheapest few sets within tolerance, and
the few closest to the target, are then refined with SLSQP to the cheapest
concentrations whose CIEDE2000 stays within tolerance. The result is fully
deterministic.

Units. Concentration is dimensionless: 1.0 is the depth at which the
catalogue colour was defined, and the caller says how many grams of dye per
kg of fabric that is. ``unit_cost`` is the cost of one unit of concentration
per kg of fabric, so a formulation's cost is per kg of fabric.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from costing.color import perceptual as p

WHITE = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class Dye:
    name: str
    reflectance: tuple[float, float, float]  # sRGB-encoded, in [0, 1], at unit concentration
    unit_cost: float  # cost per unit concentration per kg of fabric


@dataclass(frozen=True)
class SolverParams:
    max_dyes: int
    max_concentration: float
    min_concentration: float
    polish_top_k: int
    polish_maxiter: int
    polish_ftol: float
    polish_margin: float


@dataclass(frozen=True)
class DyeDose:
    name: str
    concentration: float
    concentration_g_per_kg: float
    cost: float


@dataclass(frozen=True)
class Formulation:
    dyes: tuple[DyeDose, ...]
    cost: float
    delta_e: float
    feasible: bool
    convergence_status: str  # "converged" | "max_iterations" (of the final SLSQP refinement)
    supports_evaluated: int
    predicted_srgb: tuple[float, float, float]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["concentration_g_per_kg"] = {x.name: x.concentration_g_per_kg for x in self.dyes}
        return d


def predict(conc: np.ndarray, ks_dyes: np.ndarray, ks_substrate: np.ndarray) -> np.ndarray:
    """Kubelka-Munk mixture reflectance for each row of conc, shape (P, 3)."""
    return np.clip(p.ks_to_reflectance(ks_substrate + conc @ ks_dyes), 0.0, 1.0)


def rank_keys(feasible: np.ndarray, cost: np.ndarray, de: np.ndarray) -> np.ndarray:
    """Indices ordered by feasibility rules: feasible by cost, then infeasible by delta E."""
    return np.lexsort((np.where(feasible, cost, de), np.where(feasible, 0, 1)))


def enumerate_supports(n_dyes: int, max_dyes: int) -> list[tuple[int, ...]]:
    """All dye index sets of size 1..max_dyes."""
    return [s for k in range(1, max_dyes + 1) for s in combinations(range(n_dyes), k)]


def screen(
    supports: list[tuple[int, ...]],
    ks_dyes: np.ndarray,
    ks_target_excess: np.ndarray,
    cmax: float,
) -> np.ndarray:
    """Least-squares concentrations in K/S space for every support, shape (S, n_dyes).

    Solves ks_dyes[support].T @ c = ks_target_excess per support, clipped to
    [0, cmax]. Exact when the support has 3 dyes and the target is reachable.
    """
    out = np.zeros((len(supports), ks_dyes.shape[0]))
    by_size: dict[int, list[int]] = {}
    for row, s in enumerate(supports):
        by_size.setdefault(len(s), []).append(row)
    for rows in by_size.values():
        idx = np.array([supports[r] for r in rows])  # (m, k)
        a = np.transpose(ks_dyes[idx], (0, 2, 1))  # (m, 3, k)
        c = np.einsum("mkj,j->mk", np.linalg.pinv(a), ks_target_excess)
        block = np.zeros((len(rows), ks_dyes.shape[0]))
        np.put_along_axis(block, idx, np.clip(c, 0.0, cmax), axis=1)
        out[rows] = block
    return out


def polish(
    start: np.ndarray,
    support: tuple[int, ...],
    ks_dyes: np.ndarray,
    ks_sub: np.ndarray,
    unit_cost: np.ndarray,
    target_lab: np.ndarray,
    tolerance: float,
    params: SolverParams,
) -> tuple[np.ndarray, bool]:
    """Cheapest concentrations on support with delta E <= tolerance - margin (SLSQP).

    Returns (concentrations, converged).
    """
    idx = list(support)

    def full(x: np.ndarray) -> np.ndarray:
        c = np.zeros_like(start)
        c[idx] = x
        return c

    def slack(x: np.ndarray) -> float:
        pred = predict(full(x)[None, :], ks_dyes, ks_sub)
        de = p.delta_e_2000(p.srgb_to_lab(pred), target_lab)[0]
        return float(tolerance - params.polish_margin - de)

    res = minimize(
        lambda x: float(unit_cost[idx] @ x),
        start[idx],
        method="SLSQP",
        bounds=[(0.0, params.max_concentration)] * len(idx),
        constraints=[{"type": "ineq", "fun": slack}],
        options={"maxiter": params.polish_maxiter, "ftol": params.polish_ftol},
    )
    out = full(np.clip(res.x, 0.0, params.max_concentration))
    out[out < params.min_concentration] = 0.0
    return out, bool(res.success)


def solve(
    target_srgb: np.ndarray | tuple[float, float, float],
    catalogue: list[Dye],
    *,
    params: SolverParams,
    tolerance: float,
    reference_depth_g_per_kg: float,
    substrate: np.ndarray | tuple[float, float, float] = WHITE,
) -> Formulation:
    """Cheapest formulation whose predicted colour is within tolerance of target.

    target_srgb and substrate are sRGB in [0, 1]. If no formulation is within
    tolerance the closest one found is returned with feasible=False.
    """
    target = np.asarray(target_srgb, dtype=np.float64)
    target_lab = p.srgb_to_lab(target)
    ks_dyes = p.reflectance_to_ks(np.array([d.reflectance for d in catalogue]))
    ks_sub = p.reflectance_to_ks(np.asarray(substrate, dtype=np.float64))
    unit_cost = np.array([d.unit_cost for d in catalogue])

    def evaluate(pop: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        de = p.delta_e_2000(p.srgb_to_lab(predict(pop, ks_dyes, ks_sub)), target_lab)
        return de <= tolerance, pop @ unit_cost, de

    supports = enumerate_supports(len(catalogue), min(params.max_dyes, len(catalogue)))
    screened = screen(
        supports, ks_dyes, p.reflectance_to_ks(target) - ks_sub, params.max_concentration
    )
    screened[screened < params.min_concentration] = 0.0
    feasible, cost, de = evaluate(screened)

    by_rank = rank_keys(feasible, cost, de)[: params.polish_top_k]
    by_de = np.argsort(de, kind="stable")[: params.polish_top_k]
    chosen = list(dict.fromkeys([*by_rank.tolist(), *by_de.tolist()]))

    candidates = [screened[r] for r in chosen]
    converged = [True] * len(candidates)
    for r in chosen:
        c, ok = polish(
            screened[r], supports[r], ks_dyes, ks_sub, unit_cost, target_lab, tolerance, params
        )
        candidates.append(c)
        converged.append(ok)
    cand = np.array(candidates)
    f2, c2, d2 = evaluate(cand)
    i = int(rank_keys(f2, c2, d2)[0])
    best = cand[i]

    doses = tuple(
        DyeDose(
            name=catalogue[j].name,
            concentration=float(best[j]),
            concentration_g_per_kg=float(best[j] * reference_depth_g_per_kg),
            cost=float(best[j] * unit_cost[j]),
        )
        for j in np.argsort(-best, kind="stable")
        if best[j] > 0
    )
    return Formulation(
        dyes=doses,
        cost=float(c2[i]),
        delta_e=float(d2[i]),
        feasible=bool(f2[i]),
        convergence_status="converged" if converged[i] else "max_iterations",
        supports_evaluated=len(supports),
        predicted_srgb=tuple(float(x) for x in predict(best[None, :], ks_dyes, ks_sub)[0]),
    )
