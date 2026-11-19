"""Specification tests for CIEDE2000 and its contrast against raw sRGB
Euclidean distance (the incumbent's distance metric).

See tests/color_science/README.md for the test-naming convention.
"""

import colour
import numpy as np
import pytest

from costing.color import perceptual as p

# Sharma, Wu & Dalal (2005), "The CIEDE2000 Color-Difference Formula:
# Implementation Notes, Supplementary Test Data, and Mathematical
# Observations". Reproduced to within 1e-4 by this implementation.
SHARMA_PAIRS = [
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 0.0000, -2.5000), 4.3065),
    ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
]


@pytest.mark.parametrize("lab1, lab2, expected", SHARMA_PAIRS)
def test_de2000_sharma_reference_vectors(lab1, lab2, expected):
    de = p.delta_e_2000(np.array(lab1), np.array(lab2))
    assert de == pytest.approx(expected, abs=1e-4)


def test_de2000_hue_wraparound():
    # Hues of 1.91 degrees and 358.09 degrees are 3.82 degrees apart on the
    # circle, not ~356 apart. A naive arithmetic mean hue returns 1.3644, an
    # 11% error against the correct 1.2280 — tight enough to catch that bug.
    de = p.delta_e_2000(np.array([50.0, 30.0, 1.0]), np.array([50.0, 30.0, -1.0]))
    assert de == pytest.approx(1.2280, abs=1e-3)


def test_de2000_triangle_inequality_is_violated():
    # This is load-bearing, not a curiosity: it rules out metric-tree
    # indexing (KD-tree, ball tree, VP-tree) over the dye catalogue, because
    # they all rely on the triangle inequality for pruning. See ADR-004.
    a = np.array([34.5, 59.5, -41.0])
    b = np.array([42.6, 9.4, -44.1])
    c = np.array([52.5, -58.1, -86.9])
    d_ac = p.delta_e_2000(a, c)
    d_ab = p.delta_e_2000(a, b)
    d_bc = p.delta_e_2000(b, c)
    assert d_ac == pytest.approx(83.92, abs=0.05)
    assert (d_ab + d_bc) == pytest.approx(64.34, abs=0.05)
    assert d_ac > d_ab + d_bc


def test_de2000_identity_zero():
    lab = np.array([45.0, 12.0, -30.0])
    assert p.delta_e_2000(lab, lab) == pytest.approx(0.0, abs=1e-9)


def test_de2000_symmetry():
    rng = np.random.default_rng(10)
    lab1 = np.column_stack(
        [rng.uniform(0, 100, 500), rng.uniform(-100, 100, 500), rng.uniform(-100, 100, 500)]
    )
    lab2 = np.column_stack(
        [rng.uniform(0, 100, 500), rng.uniform(-100, 100, 500), rng.uniform(-100, 100, 500)]
    )
    forward = p.delta_e_2000(lab1, lab2)
    backward = p.delta_e_2000(lab2, lab1)
    assert np.allclose(forward, backward, atol=1e-9)


def test_de2000_non_negative_and_finite():
    rng = np.random.default_rng(11)
    lab1 = np.column_stack(
        [rng.uniform(0, 100, 2000), rng.uniform(-100, 100, 2000), rng.uniform(-100, 100, 2000)]
    )
    lab2 = np.column_stack(
        [rng.uniform(0, 100, 2000), rng.uniform(-100, 100, 2000), rng.uniform(-100, 100, 2000)]
    )
    de = p.delta_e_2000(lab1, lab2)
    assert np.all(np.isfinite(de))
    assert np.all(de >= 0.0)


def test_de2000_neutral_samples_produce_no_nan():
    neutrals = np.column_stack(
        [np.linspace(0, 100, 50), np.zeros(50), np.zeros(50)]
    )
    de = p.delta_e_2000(neutrals[:-1], neutrals[1:])
    assert not np.any(np.isnan(de))


def test_de2000_kL_halves_pure_lightness_difference():
    lab1 = np.array([50.0, 20.0, 10.0])
    lab2 = np.array([60.0, 20.0, 10.0])
    de_default = p.delta_e_2000(lab1, lab2, k_L=1.0)
    de_kl2 = p.delta_e_2000(lab1, lab2, k_L=2.0)
    assert de_kl2 == pytest.approx(de_default / 2.0, abs=1e-9)


def test_de2000_kL_leaves_pure_chroma_difference_untouched():
    hue = np.radians(30.0)
    lab1 = np.array([50.0, 20.0 * np.cos(hue), 20.0 * np.sin(hue)])
    lab2 = np.array([50.0, 30.0 * np.cos(hue), 30.0 * np.sin(hue)])
    de_default = p.delta_e_2000(lab1, lab2, k_L=1.0)
    de_kl2 = p.delta_e_2000(lab1, lab2, k_L=2.0)
    assert de_kl2 == pytest.approx(de_default, abs=1e-9)


def test_rgb_equal_distance_spans_25x_perceptual_difference():
    # A fixed sRGB step of [0.05, 0, 0] from five anchors is IDENTICAL in
    # sRGB distance every time. In ΔE2000 it ranges from imperceptible to a
    # QC reject — the incumbent's distance metric cannot tell these apart.
    anchors = {
        "mid_green": np.array([0.00, 0.55, 0.00]),
        "dark_blue": np.array([0.00, 0.00, 0.35]),
        "saturated_yellow": np.array([0.90, 0.85, 0.00]),
        "light_grey": np.array([0.75, 0.75, 0.75]),
        "dark_grey": np.array([0.10, 0.10, 0.10]),
    }
    step = np.array([0.05, 0.0, 0.0])

    rgb_distances = []
    de_values = {}
    for name, anchor in anchors.items():
        other = np.clip(anchor + step, 0.0, 1.0)
        rgb_distances.append(p.rgb_euclidean(anchor, other))
        de_values[name] = float(
            p.delta_e_2000(p.srgb_to_lab(anchor), p.srgb_to_lab(other))
        )

    assert len(set(np.round(rgb_distances, 6))) == 1

    assert de_values["mid_green"] == pytest.approx(0.323, abs=0.005)
    assert de_values["dark_blue"] == pytest.approx(0.978, abs=0.005)
    assert de_values["saturated_yellow"] == pytest.approx(3.029, abs=0.005)
    assert de_values["light_grey"] == pytest.approx(6.016, abs=0.005)
    assert de_values["dark_grey"] == pytest.approx(8.132, abs=0.005)

    assert min(de_values.values()) < 1.0
    assert max(de_values.values()) > 8.0
    assert max(de_values.values()) / min(de_values.values()) > 20.0


def test_rgb_distance_ranks_dye_candidates_incorrectly():
    # The one that costs money: a solver minimising sRGB distance prefers
    # candidate A (closer in sRGB) over candidate B, even though B is the
    # perceptually closer, textile-tolerance-passing match.
    target_a = np.array([0.10, 0.10, 0.10])
    candidate_a = np.array([0.15, 0.10, 0.10])
    target_b = np.array([0.00, 0.55, 0.00])
    candidate_b = np.array([0.20, 0.55, 0.00])

    rgb_dist_a = p.rgb_euclidean(target_a, candidate_a)
    rgb_dist_b = p.rgb_euclidean(target_b, candidate_b)
    de_a = p.delta_e_2000(p.srgb_to_lab(target_a), p.srgb_to_lab(candidate_a))
    de_b = p.delta_e_2000(p.srgb_to_lab(target_b), p.srgb_to_lab(candidate_b))

    assert rgb_dist_a == pytest.approx(0.05, abs=1e-6)
    assert rgb_dist_b == pytest.approx(0.20, abs=1e-6)
    assert de_a == pytest.approx(8.132, abs=0.005)
    assert de_b == pytest.approx(2.680, abs=0.005)

    # B is ~4x further in sRGB but B is the perceptually closer match.
    assert rgb_dist_b / rgb_dist_a == pytest.approx(4.0, abs=0.01)
    assert de_a / de_b > 3.0

    # An sRGB-minimising solver picks A; the perceptually correct pick is B.
    assert rgb_dist_a < rgb_dist_b
    assert de_a > de_b


def test_rgb_distance_has_no_interpretable_acceptance_threshold():
    # The same sRGB distance (0.05) yields contradictory pass/fail verdicts
    # against a ΔE <= 2 textile tolerance, depending on which colour region
    # it's measured in — so no QC spec can be written in sRGB units.
    tolerance = 2.0
    pairs = [
        (np.array([0.00, 0.55, 0.00]), np.array([0.05, 0.55, 0.00])),  # passes
        (np.array([0.10, 0.10, 0.10]), np.array([0.15, 0.10, 0.10])),  # fails
    ]
    verdicts = []
    for a, b in pairs:
        assert p.rgb_euclidean(a, b) == pytest.approx(0.05, abs=1e-6)
        de = p.delta_e_2000(p.srgb_to_lab(a), p.srgb_to_lab(b))
        verdicts.append(de <= tolerance)
    assert verdicts[0] != verdicts[1]


def test_de76_is_an_improvement_but_not_sufficient():
    # CIE76 (plain Euclidean distance in Lab) is closer to perceptual
    # reality than raw sRGB distance, but it over-weights high-chroma
    # regions relative to CIEDE2000 — and fashion textiles live in
    # high-chroma regions. This rules out the cheap half-migration of
    # switching to CIE76 without the full CIEDE2000 weighting functions.
    step = np.array([0.05, 0.0, 0.0])

    saturated = np.array([0.00, 0.55, 0.00])  # high chroma
    neutral = np.array([0.10, 0.10, 0.10])  # near-achromatic

    def ratio(anchor):
        other = np.clip(anchor + step, 0.0, 1.0)
        lab1, lab2 = p.srgb_to_lab(anchor), p.srgb_to_lab(other)
        return p.delta_e_76(lab1, lab2) / p.delta_e_2000(lab1, lab2)

    saturated_ratio = ratio(saturated)
    neutral_ratio = ratio(neutral)

    # CIE76 is not just a scaled version of CIE2000 (an "improvement" would
    # be a metric that at least agrees on relative magnitudes); the bias is
    # much stronger in the saturated region than in the near-neutral one.
    assert saturated_ratio > 2.0
    assert neutral_ratio < 1.2
    assert saturated_ratio > 2.0 * neutral_ratio


def test_de2000_matches_colour_science_oracle():
    rng = np.random.default_rng(12)
    lab1 = np.column_stack(
        [rng.uniform(20, 90, 300), rng.uniform(-80, 80, 300), rng.uniform(-80, 80, 300)]
    )
    lab2 = np.column_stack(
        [rng.uniform(20, 90, 300), rng.uniform(-80, 80, 300), rng.uniform(-80, 80, 300)]
    )
    ours = p.delta_e_2000(lab1, lab2)
    oracle = colour.delta_E(lab1, lab2, method="CIE 2000")
    assert np.max(np.abs(ours - oracle)) < 1e-4
