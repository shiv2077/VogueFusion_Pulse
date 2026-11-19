"""Specification tests for single-constant Kubelka-Munk subtractive mixing,
contrasted against the incumbent's concentration-weighted sRGB averaging.

See tests/color_science/README.md for the test-naming convention.
"""

import numpy as np
import pytest

from costing.color import perceptual as p

CYAN_BLUE = np.array([0.10, 0.20, 0.80])
YELLOW = np.array([0.85, 0.75, 0.08])
MAGENTA = np.array([0.80, 0.10, 0.45])
BLACK = np.array([0.04, 0.04, 0.04])
COTTON_SUBSTRATE = np.array([0.82, 0.80, 0.75])

ALL_DYES = [CYAN_BLUE, YELLOW, MAGENTA, BLACK]


def test_km_blue_and_yellow_make_green_not_grey():
    # The headline case. Equal parts (unit concentration each) of a cyan-
    # leaning blue and a yellow:
    #
    #   model              result                      L*      a*      chroma / hue
    #   Kubelka-Munk       [0.0997, 0.1966, 0.0797]     18.17  -16.86   23.2 @ 136.6 deg (green)
    #   RGB averaging      [0.4750, 0.4750, 0.4400]     50.63   -1.73    5.19 (near-neutral)
    #
    # ΔE2000 between the two predictions is 30.0. Textile QC rejects at 2:
    # the incumbent model is off by fifteen rejection thresholds on the most
    # elementary mixture in the catalogue.
    km_result = p.mix_subtractive(np.array([CYAN_BLUE, YELLOW]), np.array([1.0, 1.0]))
    rgb_result = p.mix_additive_rgb(np.array([CYAN_BLUE, YELLOW]), np.array([0.5, 0.5]))

    km_lab = p.srgb_to_lab(km_result)
    rgb_lab = p.srgb_to_lab(rgb_result)

    assert km_result == pytest.approx([0.0997, 0.1966, 0.0797], abs=1e-3)
    assert km_lab[0] == pytest.approx(18.17, abs=0.05)
    assert km_lab[1] == pytest.approx(-16.86, abs=0.05)
    km_chroma = float(np.hypot(km_lab[1], km_lab[2]))
    km_hue = float(np.degrees(np.arctan2(km_lab[2], km_lab[1])) % 360.0)
    assert km_chroma == pytest.approx(23.2, abs=0.1)
    assert km_hue == pytest.approx(136.6, abs=0.1)

    assert rgb_result == pytest.approx([0.475, 0.475, 0.44], abs=1e-3)
    rgb_chroma = float(np.hypot(rgb_lab[1], rgb_lab[2]))
    assert rgb_chroma < 6.0  # near-neutral, unlike the KM prediction

    de = p.delta_e_2000(km_lab, rgb_lab)
    assert de == pytest.approx(30.0, abs=0.1)
    assert de > 15 * 2.0  # 15x the textile QC tolerance of ΔE 2


def test_rgb_averaging_cannot_produce_a_mixture_darker_than_its_inputs():
    # A weighted mean is bounded below by its inputs (per channel), so RGB
    # averaging can never predict that two mid-tones yield a dark shade.
    # This is a structural impossibility of the model, not a tuning problem
    # -- no parameter fitting rescues it. Subtractive mixing does this
    # constantly; it is how navy and charcoal are made.
    dyes = np.array([CYAN_BLUE, MAGENTA])
    floor = np.minimum(CYAN_BLUE, MAGENTA)

    additive = p.mix_additive_rgb(dyes, np.array([0.5, 0.5]))
    subtractive = p.mix_subtractive(dyes, np.array([1.0, 1.0]))

    assert np.all(additive >= floor - 1e-9)
    assert np.any(subtractive < floor)


def test_km_roundtrip():
    rng = np.random.default_rng(20)
    reflectance = rng.uniform(1e-3, 1.0, size=500)
    ks = p.reflectance_to_ks(reflectance)
    roundtrip = p.ks_to_reflectance(ks)
    assert np.max(np.abs(roundtrip - reflectance)) < 1e-9


def test_km_zero_for_perfect_diffuser():
    assert p.reflectance_to_ks(np.array(1.0)) == pytest.approx(0.0, abs=1e-9)


def test_km_monotonically_increasing_as_reflectance_falls():
    reflectance = np.linspace(0.001, 1.0, 500)
    ks = p.reflectance_to_ks(reflectance)
    assert np.all(np.diff(ks) < 0)  # increasing reflectance -> decreasing K/S


def test_km_finite_at_r_zero():
    ks = p.reflectance_to_ks(np.array(0.0))
    assert np.isfinite(ks)


def test_zero_concentration_returns_substrate():
    result = p.mix_subtractive(
        np.array([CYAN_BLUE]), np.array([0.0]), substrate_reflectance=COTTON_SUBSTRATE
    )
    assert result == pytest.approx(COTTON_SUBSTRATE, abs=1e-9)


def test_unit_concentration_single_dye_reproduces_dye():
    result = p.mix_subtractive(np.array([CYAN_BLUE]), np.array([1.0]))
    assert result == pytest.approx(CYAN_BLUE, abs=1e-9)


def test_mixing_commutative():
    a = p.mix_subtractive(np.array([CYAN_BLUE, YELLOW]), np.array([0.7, 1.3]))
    b = p.mix_subtractive(np.array([YELLOW, CYAN_BLUE]), np.array([1.3, 0.7]))
    assert a == pytest.approx(b, abs=1e-9)


@pytest.mark.parametrize("dye_index", [0, 1, 2, 3])
def test_more_dye_always_darker(dye_index):
    # If this fails, the genetic algorithm can "save money" by adding more
    # dye and getting a lighter result -- a search runs away chasing a
    # physically nonsensical gradient.
    dye = ALL_DYES[dye_index]
    loads = np.array([0.1, 0.5, 1.0, 2.0, 3.5, 5.0])
    reflectances = np.array(
        [p.mix_subtractive(np.array([dye]), np.array([load])) for load in loads]
    )
    luminance = reflectances @ np.array([0.2126729, 0.7151522, 0.0721750])
    assert np.all(np.diff(luminance) < 0)


def test_reflectance_stays_in_unit_interval():
    rng = np.random.default_rng(21)
    for _ in range(2000):
        concentrations = rng.uniform(0.0, 5.0, size=4)
        result = p.mix_subtractive(np.array(ALL_DYES), concentrations)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


def test_negative_concentration_raises():
    with pytest.raises(ValueError, match="non-negative"):
        p.mix_subtractive(np.array([CYAN_BLUE]), np.array([-0.1]))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="mismatch"):
        p.mix_subtractive(np.array([CYAN_BLUE, YELLOW]), np.array([1.0]))


def test_substrate_divergence_at_pale_load():
    # At pale load, substrate choice alone exceeds textile tolerance (ΔE 2)
    # -- this is exactly the K/S normalisation trap described in ADR-005:
    # defining the dye constant as an excess over the substrate it sits on
    # makes predictions on different substrates diverge without bound.
    load = 0.15
    ideal_white = p.mix_subtractive(np.array([CYAN_BLUE]), np.array([load]))
    cotton = p.mix_subtractive(
        np.array([CYAN_BLUE]), np.array([load]), substrate_reflectance=COTTON_SUBSTRATE
    )
    de = p.delta_e_2000(p.srgb_to_lab(ideal_white), p.srgb_to_lab(cotton))
    assert de > 2.0


def test_substrate_gap_shrinks_monotonically_with_load():
    loads = [0.2, 1.0, 5.0, 20.0]
    gaps = []
    for load in loads:
        ideal_white = p.mix_subtractive(np.array([CYAN_BLUE]), np.array([load]))
        cotton = p.mix_subtractive(
            np.array([CYAN_BLUE]), np.array([load]), substrate_reflectance=COTTON_SUBSTRATE
        )
        gaps.append(
            float(p.delta_e_2000(p.srgb_to_lab(ideal_white), p.srgb_to_lab(cotton)))
        )
    assert all(earlier > later for earlier, later in zip(gaps, gaps[1:], strict=False))
