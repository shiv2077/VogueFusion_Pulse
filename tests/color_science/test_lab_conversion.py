"""Specification tests for sRGB <-> linear <-> XYZ <-> Lab conversions.

See tests/color_science/README.md for the test-naming convention and the
scope of this suite (a migration specification, not a regression suite for
the live pipeline).
"""

import colour
import numpy as np
import pytest

from costing.color import perceptual as p


def test_grey_lightness_verified_constant():
    # Verified constant: mid-grey sRGB 128/255 has L* = 53.585, not 50 — the
    # sRGB EOTF is not linear, so "half the code value" is not "half the
    # lightness".
    lab = p.srgb_to_lab(np.array([128 / 255] * 3))
    assert lab[0] == pytest.approx(53.585, abs=0.01)


def test_neutral_ab_error_floor():
    # A perfectly neutral colour (R=G=B) should map to a*=b*=0. In practice
    # the sRGB primaries matrix and the D65 white point constants carry a
    # small mismatch, leaving an empirical residual floor around 1.67e-5.
    # This is a property of the standard constants, not a bug in the
    # conversion, so the tolerance is deliberately loose.
    for value in (0.0, 0.1, 0.5, 0.75, 1.0):
        lab = p.srgb_to_lab(np.array([value] * 3))
        assert lab[1] == pytest.approx(0.0, abs=1e-4)
        assert lab[2] == pytest.approx(0.0, abs=1e-4)


def test_eotf_breakpoint_discontinuity():
    # The IEC 61966-2-1 EOTF is defined piecewise, and the two pieces do not
    # meet exactly at the breakpoint because the published constants (12.92,
    # 0.055, 2.4, 0.04045, 0.0031308) are themselves rounded. The resulting
    # gap (~2.5e-9) is the standard's residual, not an implementation bug.
    below = 0.04045
    above = np.nextafter(below, 1.0)
    gap = abs(float(p.srgb_to_linear(above) - p.srgb_to_linear(below)))
    assert gap < 1e-8


def test_srgb_linear_roundtrip_lossless():
    rng = np.random.default_rng(0)
    samples = rng.uniform(0.0, 1.0, size=(2000, 3))
    roundtrip = p.linear_to_srgb(p.srgb_to_linear(samples))
    assert np.max(np.abs(roundtrip - samples)) < 1e-9


def test_lab_xyz_roundtrip():
    rng = np.random.default_rng(1)
    lab = np.column_stack(
        [
            rng.uniform(0.0, 100.0, 2000),
            rng.uniform(-100.0, 100.0, 2000),
            rng.uniform(-100.0, 100.0, 2000),
        ]
    )
    roundtrip = p.xyz_to_lab(p.lab_to_xyz(lab))
    assert np.max(np.abs(roundtrip - lab)) < 1e-9


def test_lab_xyz_roundtrip_dark_segment():
    # Dark colours (low L*) exercise the Lab->XYZ linear-segment branch for
    # Y, which is keyed on L* directly rather than on fy**3. This is exactly
    # where navy and black dye formulations live, so it gets its own test.
    rng = np.random.default_rng(2)
    lab = np.column_stack(
        [
            rng.uniform(0.0, 12.0, 200),
            rng.uniform(-50.0, 50.0, 200),
            rng.uniform(-50.0, 50.0, 200),
        ]
    )
    roundtrip = p.xyz_to_lab(p.lab_to_xyz(lab))
    assert np.max(np.abs(roundtrip - lab)) < 1e-9


def test_srgb_xyz_roundtrip():
    rng = np.random.default_rng(3)
    samples = rng.uniform(0.0, 1.0, size=(500, 3))
    roundtrip = p.xyz_to_srgb(p.srgb_to_xyz(samples))
    assert np.max(np.abs(roundtrip - samples)) < 1e-9


def test_srgb_lab_full_pipeline_roundtrip():
    rng = np.random.default_rng(4)
    samples = rng.uniform(0.0, 1.0, size=(500, 3))
    roundtrip = p.lab_to_srgb(p.srgb_to_lab(samples))
    assert np.max(np.abs(roundtrip - samples)) < 1e-8


def test_lightness_monotonic_in_luminance():
    greys = np.linspace(0.0, 1.0, 256)
    rgb = np.column_stack([greys, greys, greys])
    lab = p.srgb_to_lab(rgb)
    assert np.all(np.diff(lab[:, 0]) > 0)


@pytest.mark.parametrize(
    "name, rgb, channel, sign",
    [
        ("red_is_plus_a", [0.8, 0.1, 0.1], 1, 1),
        ("green_is_minus_a", [0.1, 0.8, 0.1], 1, -1),
        ("yellow_is_plus_b", [0.8, 0.8, 0.1], 2, 1),
        ("blue_is_minus_b", [0.1, 0.1, 0.8], 2, -1),
    ],
)
def test_axis_sign_convention(name, rgb, channel, sign):
    lab = p.srgb_to_lab(np.array(rgb))
    assert np.sign(lab[channel]) == sign


def test_out_of_gamut_lab_clips_not_wraps():
    # An Lab value with no valid sRGB representation should clip into
    # [0, 1], not wrap around (e.g. via modulo) into an unrelated colour.
    out_of_gamut = np.array([50.0, 200.0, 200.0])
    srgb = p.lab_to_srgb(out_of_gamut)
    assert np.all(srgb >= 0.0)
    assert np.all(srgb <= 1.0)


def test_conversions_vectorized_over_batch():
    rng = np.random.default_rng(5)
    batch = rng.uniform(0.0, 1.0, size=(50, 3))
    lab_batch = p.srgb_to_lab(batch)
    assert lab_batch.shape == (50, 3)
    for i in range(50):
        assert np.allclose(lab_batch[i], p.srgb_to_lab(batch[i]), atol=1e-9)


def test_black_and_white_extremes():
    black = p.srgb_to_lab(np.array([0.0, 0.0, 0.0]))
    white = p.srgb_to_lab(np.array([1.0, 1.0, 1.0]))
    assert black[0] == pytest.approx(0.0, abs=1e-6)
    assert white[0] == pytest.approx(100.0, abs=1e-4)
    assert white[1] == pytest.approx(0.0, abs=1e-4)
    assert white[2] == pytest.approx(0.0, abs=1e-4)


def test_lab_matches_colour_science_oracle():
    # colour-science is a test-only oracle (ADR-006): it is never imported
    # by perceptual.py, only by this test suite, to cross-check the
    # reference implementation against an independent source.
    rng = np.random.default_rng(6)
    samples = rng.uniform(0.0, 1.0, size=(100, 3))
    ours = p.srgb_to_lab(samples)
    oracle = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(samples))
    assert np.max(np.abs(ours - oracle)) < 0.05
