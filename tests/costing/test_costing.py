"""Tests for catalogue loading, colour extraction and the quote.

The three test_regression_* tests each pin one bug in the original Streamlit
cost path (trend_and_cost_prediction/pages/4 and 5).
"""

import io
import json

import numpy as np
import pytest
from PIL import Image

from costing import solver as s
from costing.catalogue import CatalogueEntry, load_entries, to_dyes, unit_cost
from costing.color import perceptual as p
from costing.extract import Colour, dominant_colours, load_pixels
from costing.quote import Fabric, colour_line, fabric_mass_kg, quote
from voguefusion import config

CFG = config.load()
PARAMS = config.solver_params(CFG)
DEPTH = CFG["costing"]["reference_depth_g_per_kg"]
TOL = CFG["costing"]["delta_e_tolerance"]

ENTRIES = [
    CatalogueEntry("BLUE", (25, 50, 205), 400.0),
    CatalogueEntry("YELLOW", (215, 190, 20), 150.0),
    CatalogueEntry("RED", (205, 25, 115), 500.0),
    CatalogueEntry("BLACK", (10, 10, 10), 130.0),
]
DYES = to_dyes(ENTRIES, DEPTH, 0.03, 0.95)
WHITE_FABRIC = Fabric(gsm=100, length_m=10, width_m=1, substrate_srgb=(1.0, 1.0, 1.0))
# A green reachable by construction: 0.3 BLUE + 0.4 YELLOW under the solver's own model.
GREEN = tuple(
    float(x)
    for x in p.mix_subtractive(
        np.array([DYES[0].reflectance, DYES[1].reflectance]), np.array([0.3, 0.4])
    )
)


def _png(pixels: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_unit_cost_is_price_times_depth():
    assert unit_cost(500.0, 20.0) == pytest.approx(10.0)


def test_load_entries_reads_repo_catalogue():
    entries = load_entries(config.repo_path(CFG["costing"]["catalogue_file"]))
    assert len(entries) == 20
    assert all(e.price_per_kg > 0 for e in entries)


def test_to_dyes_clamps_swatch_reflectance():
    (dye,) = to_dyes([CatalogueEntry("K", (0, 255, 128), 1.0)], DEPTH, 0.03, 0.95)
    assert dye.reflectance[0] == pytest.approx(0.03)
    assert dye.reflectance[1] == pytest.approx(0.95)


def test_fabric_mass_closed_form():
    assert fabric_mass_kg(Fabric(70, 5.5, 1.1, (1, 1, 1))) == pytest.approx(70 * 5.5 * 1.1 / 1000)


def test_load_pixels_downscales_and_normalises():
    px = load_pixels(_png(np.full((400, 200, 3), 255)), max_side_px=100)
    assert px.shape == (100 * 50, 3)
    assert px.max() == pytest.approx(1.0)


def test_dominant_colours_recovers_two_flat_regions():
    img = np.zeros((40, 40, 3))
    img[:, :30] = (200, 30, 30)
    img[:, 30:] = (20, 40, 180)
    colours = dominant_colours(load_pixels(_png(img), 64), n_colours=2, n_init=2, seed=0)
    assert [round(c.fraction, 2) for c in colours] == [0.75, 0.25]
    assert colours[0].hex == "#c81e1e"
    assert colours[1].hex == "#1428b4"


def test_dominant_colours_handles_fewer_distinct_colours_than_requested():
    colours = dominant_colours(np.full((10, 3), 0.5), n_colours=5, n_init=1, seed=0)
    assert len(colours) == 1 and colours[0].fraction == 1.0


def test_regression_returns_best_candidate_seen():
    """Bug 1: the old GA read its best index from the wrong population.

    The solver's answer must be at least as cheap as every feasible candidate
    it screened.
    """
    target = np.array(GREEN)
    ks = p.reflectance_to_ks(np.array([d.reflectance for d in DYES]))
    ks_sub = p.reflectance_to_ks(np.ones(3))
    supports = s.enumerate_supports(len(DYES), PARAMS.max_dyes)
    screened = s.screen(
        supports, ks, p.reflectance_to_ks(target) - ks_sub, PARAMS.max_concentration
    )
    de = p.delta_e_2000(p.srgb_to_lab(s.predict(screened, ks, ks_sub)), p.srgb_to_lab(target))
    feasible_costs = (screened @ np.array([d.unit_cost for d in DYES]))[de <= TOL]
    result = s.solve(target, DYES, params=PARAMS, tolerance=TOL, reference_depth_g_per_kg=DEPTH)
    assert result.feasible and feasible_costs.size > 0
    assert result.cost <= feasible_costs.min() + 1e-9


def test_regression_each_colour_gets_its_own_formulation():
    """Bug 2: the old cost page applied the last colour's mixture to every colour."""
    colours = [Colour((0.30, 0.40, 0.85), 0.5), Colour((0.90, 0.85, 0.30), 0.5)]
    q = quote(
        colours,
        ENTRIES,
        DYES,
        WHITE_FABRIC,
        params=PARAMS,
        tolerance=TOL,
        reference_depth_g_per_kg=DEPTH,
        currency="INR",
        solver_version="t",
    )
    blue, yellow = q.colours
    assert blue.dyes[0].name == "BLUE"
    assert yellow.dyes[0].name == "YELLOW"
    assert q.total_cost == pytest.approx(blue.cost + yellow.cost)


def test_regression_dye_names_and_prices_stay_aligned():
    """Bug 3: the old pages zipped sorted, filtered proportions against the unsorted dye list.

    Every costed line must use the price of the dye it names.
    """
    prices = {e.name: e.price_per_kg for e in ENTRIES}
    formulation = s.solve(GREEN, DYES, params=PARAMS, tolerance=TOL, reference_depth_g_per_kg=DEPTH)
    mass = fabric_mass_kg(WHITE_FABRIC)
    line = colour_line(Colour(GREEN, 0.4), formulation, prices, mass)
    assert len(line.dyes) >= 2
    for d in line.dyes:
        assert d.price_per_kg == prices[d.name]
        assert d.cost == pytest.approx(
            mass * 0.4 * d.concentration_g_per_kg / 1000 * prices[d.name]
        )
    # The fabric-level identity: line cost == covered mass * formulation cost per kg.
    assert line.cost == pytest.approx(mass * 0.4 * formulation.cost)


def test_quote_flags_infeasible_colour():
    q = quote(
        [Colour((0.0, 1.0, 1.0), 1.0)],
        ENTRIES,
        DYES,
        WHITE_FABRIC,
        params=PARAMS,
        tolerance=TOL,
        reference_depth_g_per_kg=DEPTH,
        currency="INR",
        solver_version="t",
    )
    assert q.all_feasible is False
    assert q.colours[0].feasible is False


def test_quote_as_dict_is_json_serialisable():
    q = quote(
        [Colour((0.30, 0.40, 0.85), 1.0)],
        ENTRIES,
        DYES,
        WHITE_FABRIC,
        params=PARAMS,
        tolerance=TOL,
        reference_depth_g_per_kg=DEPTH,
        currency="INR",
        solver_version="t",
    )
    assert json.loads(json.dumps(q.as_dict()))["currency"] == "INR"
