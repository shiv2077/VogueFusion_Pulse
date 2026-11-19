"""Perceptual colour science reference implementation.

This is the colour model behind the dye solver (``costing/solver.py``) and
colour extraction (``costing/extract.py``). It replaces the original
approach in ``trend_and_cost_prediction/``, which matched dyes by Euclidean
sRGB distance and predicted mixtures by concentration-weighted sRGB
averaging; both are perceptually and physically wrong (sRGB is not
perceptually uniform; dyes mix subtractively, not additively). See
``docs/DECISIONS.md`` ADR-004 (distance metric) and ADR-005 (mixing model).

It has no runtime dependency on ``colour-science``; that package is used
only as a test oracle (see ``tests/color_science/``) to cross-check this
implementation against an independent source. Everything here is plain
numpy.

Scope and known approximation (ADR-005): the subtractive mixing model here
is a **3-channel** approximation of single-constant Kubelka-Munk, applied to
the linear-light sRGB primaries rather than to per-wavelength spectral
reflectance. A correct implementation is spectral (31 bands at 10 nm across
400-700 nm, integrated against the CIE 1931 colour-matching functions and the
illuminant SPD). The 3-channel approximation captures the ordering and
qualitative behaviour of subtractive mixing correctly (which is what a dye
solver needs to stop picking visibly-wrong matches), but it will not predict
metamerism and drifts on highly saturated mixtures. Going spectral is blocked
on obtaining per-dye reflectance curves from suppliers; see the open
measurements in ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# sRGB <-> linear-light RGB (IEC 61966-2-1)
# ---------------------------------------------------------------------------

_SRGB_EOTF_BREAKPOINT = 0.04045
_LINEAR_EOTF_BREAKPOINT = 0.0031308


def srgb_to_linear(c):
    """IEC 61966-2-1 EOTF: gamma-encoded sRGB in [0, 1] -> linear-light RGB."""
    c = np.asarray(c, dtype=np.float64)
    return np.where(
        c <= _SRGB_EOTF_BREAKPOINT,
        c / 12.92,
        ((c + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(c):
    """IEC 61966-2-1 inverse EOTF: linear-light RGB -> gamma-encoded sRGB."""
    c = np.asarray(c, dtype=np.float64)
    return np.where(
        c <= _LINEAR_EOTF_BREAKPOINT,
        c * 12.92,
        1.055 * np.power(np.clip(c, 0.0, None), 1.0 / 2.4) - 0.055,
    )


# ---------------------------------------------------------------------------
# linear RGB <-> XYZ (D65), Y scaled to [0, 100]
# ---------------------------------------------------------------------------

_M_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_M_XYZ_TO_RGB = np.linalg.inv(_M_RGB_TO_XYZ)

# CIE D65, 2-degree standard observer white point, Y normalised to 100.
D65_WHITE = np.array([95.047, 100.0, 108.883], dtype=np.float64)


def srgb_to_xyz(rgb):
    """sRGB in [0, 1] -> CIE XYZ, D65, Y in [0, 100]."""
    rgb = np.asarray(rgb, dtype=np.float64)
    linear = srgb_to_linear(rgb)
    xyz = linear @ _M_RGB_TO_XYZ.T
    return xyz * 100.0


def xyz_to_srgb(xyz):
    """CIE XYZ, D65, Y in [0, 100] -> sRGB, clipped to [0, 1] (out-of-gamut clips)."""
    xyz = np.asarray(xyz, dtype=np.float64) / 100.0
    linear = xyz @ _M_XYZ_TO_RGB.T
    srgb = linear_to_srgb(np.clip(linear, 0.0, None))
    return np.clip(srgb, 0.0, 1.0)


# ---------------------------------------------------------------------------
# XYZ <-> CIE Lab (D65)
# ---------------------------------------------------------------------------

_LAB_EPS = 216.0 / 24389.0
_LAB_KAPPA = 24389.0 / 27.0


def _lab_f(t):
    t = np.asarray(t, dtype=np.float64)
    return np.where(t > _LAB_EPS, np.cbrt(t), (_LAB_KAPPA * t + 16.0) / 116.0)


def _lab_f_inv(f):
    f = np.asarray(f, dtype=np.float64)
    return np.where(f > (_LAB_EPS ** (1.0 / 3.0)), f ** 3, (116.0 * f - 16.0) / _LAB_KAPPA)


def xyz_to_lab(xyz, white=D65_WHITE):
    """CIE XYZ (Y in [0, 100]) -> CIE Lab (D65)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    white = np.asarray(white, dtype=np.float64)
    xr = xyz[..., 0] / white[0]
    yr = xyz[..., 1] / white[1]
    zr = xyz[..., 2] / white[2]

    fx, fy, fz = _lab_f(xr), _lab_f(yr), _lab_f(zr)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab, white=D65_WHITE):
    """CIE Lab (D65) -> CIE XYZ (Y in [0, 100]).

    The Y channel does NOT use the generic f**3 inverse. It has its own
    linear-segment branch keyed on L* directly (not on fy**3): Y/Yn is
    ((L+16)/116)**3 when L > kappa*eps, else L/kappa. Overwriting fy**3 with
    this branch is required for lossless round-tripping of dark colours —
    exactly where navy and black dye formulations live.
    """
    lab = np.asarray(lab, dtype=np.float64)
    white = np.asarray(white, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    xr = _lab_f_inv(fx)
    zr = _lab_f_inv(fz)
    yr = np.where(L > _LAB_KAPPA * _LAB_EPS, ((L + 16.0) / 116.0) ** 3, L / _LAB_KAPPA)

    x = xr * white[0]
    y = yr * white[1]
    z = zr * white[2]
    return np.stack([x, y, z], axis=-1)


def srgb_to_lab(rgb):
    """Convenience wrapper: sRGB in [0, 1] -> CIE Lab (D65)."""
    return xyz_to_lab(srgb_to_xyz(rgb))


def lab_to_srgb(lab):
    """Convenience wrapper: CIE Lab (D65) -> sRGB in [0, 1] (out-of-gamut clips)."""
    return xyz_to_srgb(lab_to_xyz(lab))


# ---------------------------------------------------------------------------
# Colour difference metrics
# ---------------------------------------------------------------------------

def delta_e_76(lab1, lab2):
    """CIE76: plain Euclidean distance in Lab. Kept only so tests can contrast it."""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    return np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))


def rgb_euclidean(rgb1, rgb2):
    """Euclidean distance in raw sRGB. This is the incumbent's distance metric,
    kept only so tests can contrast it against delta_e_2000."""
    rgb1 = np.asarray(rgb1, dtype=np.float64)
    rgb2 = np.asarray(rgb2, dtype=np.float64)
    return np.sqrt(np.sum((rgb1 - rgb2) ** 2, axis=-1))


def delta_e_2000(lab1, lab2, k_L=1.0, k_C=1.0, k_H=1.0):
    """CIEDE2000 colour difference (CIE 142:2001).

    Two branch conditions are neutral-safe by construction:

    - The hue difference term is defined as 0 when either sample is
      achromatic (C1p * C2p == 0), rather than wrapping an undefined hue.
    - The mean hue term is h1p + h2p (not the average) in that same
      achromatic case; otherwise it three-way branches on
      abs(h1p - h2p) <= 180 and h1p + h2p < 360.

    Note: delta_e_2000 does not satisfy the triangle inequality (see
    tests/color_science/test_delta_e2000.py::test_triangle_inequality_is_violated).
    This rules out metric-tree indexing (KD-tree, ball tree, VP-tree) over the
    dye catalogue — see ADR-004.
    """
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)

    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    C_bar = (C1 + C2) / 2.0

    G = 0.5 * (1.0 - np.sqrt((C_bar ** 7) / (C_bar ** 7 + 25.0 ** 7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2

    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    # arctan2(0, 0) is conventionally 0, keep achromatic hues at 0.
    h1p = np.where((a1p == 0) & (b1 == 0), 0.0, h1p)
    h2p = np.where((a2p == 0) & (b2 == 0), 0.0, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    C_prod_zero = (C1p * C2p) == 0.0

    dhp_raw = h2p - h1p
    dhp = np.where(
        dhp_raw > 180.0, dhp_raw - 360.0, np.where(dhp_raw < -180.0, dhp_raw + 360.0, dhp_raw)
    )
    dhp = np.where(C_prod_zero, 0.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    L_bar_p = (L1 + L2) / 2.0
    C_bar_p = (C1p + C2p) / 2.0

    h_sum = h1p + h2p
    h_diff_abs = np.abs(h1p - h2p)
    h_bar_p_nonzero = np.where(
        h_diff_abs <= 180.0,
        h_sum / 2.0,
        np.where(h_sum < 360.0, (h_sum + 360.0) / 2.0, (h_sum - 360.0) / 2.0),
    )
    h_bar_p = np.where(C_prod_zero, h_sum, h_bar_p_nonzero)

    T = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_p - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_p))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_p - 63.0))
    )

    d_theta = 30.0 * np.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    R_C = 2.0 * np.sqrt((C_bar_p ** 7) / (C_bar_p ** 7 + 25.0 ** 7))
    S_L = 1.0 + (0.015 * (L_bar_p - 50.0) ** 2) / np.sqrt(20.0 + (L_bar_p - 50.0) ** 2)
    S_C = 1.0 + 0.045 * C_bar_p
    S_H = 1.0 + 0.015 * C_bar_p * T
    R_T = -np.sin(np.radians(2.0 * d_theta)) * R_C

    term_L = dLp / (k_L * S_L)
    term_C = dCp / (k_C * S_C)
    term_H = dHp / (k_H * S_H)

    return np.sqrt(term_L ** 2 + term_C ** 2 + term_H ** 2 + R_T * term_C * term_H)


# ---------------------------------------------------------------------------
# Kubelka-Munk single-constant subtractive mixing
# ---------------------------------------------------------------------------

_REFLECTANCE_FLOOR = 1e-6


def reflectance_to_ks(reflectance):
    """Single-constant Kubelka-Munk: reflectance in [0, 1] -> K/S.

    Reflectance is floored at 1e-6 before the transform: a pure black dye
    (R=0) sends K/S to infinity and poisons a GA fitness function built on
    top of this.
    """
    r = np.clip(np.asarray(reflectance, dtype=np.float64), _REFLECTANCE_FLOOR, 1.0)
    return ((1.0 - r) ** 2) / (2.0 * r)


def ks_to_reflectance(ks):
    """Inverse single-constant Kubelka-Munk: K/S -> reflectance in [0, 1]."""
    ks = np.asarray(ks, dtype=np.float64)
    return 1.0 + ks - np.sqrt(ks ** 2 + 2.0 * ks)


def mix_subtractive(dye_reflectances, concentrations, substrate_reflectance=None):
    """Predict the reflectance of a dye mixture via single-constant Kubelka-Munk.

    dye_reflectances: array-like, shape (n_dyes, n_channels) — each row is a
        dye's reflectance at unit concentration.
    concentrations: array-like, shape (n_dyes,) — non-negative concentration
        (dye load) of each dye.
    substrate_reflectance: array-like, shape (n_channels,), optional — the
        undyed substrate's reflectance. Defaults to a perfect white
        ([1, 1, 1] for the 3-channel approximation).

    The mixture K/S is the substrate's K/S plus the concentration-weighted
    sum of each dye's K/S, measured against a perfect white reference:

        ks_mix = reflectance_to_ks(substrate) + concentrations @ reflectance_to_ks(dyes)

    Each dye's K/S constant is measured against ideal white, not as an
    excess over whatever substrate it happens to sit on
    (ks_dye - ks_substrate). Defining it as an excess makes the dye constant
    substrate-dependent, and predictions on different substrates diverge
    without bound as dye load rises instead of converging (see
    tests/color_science/test_subtractive_mixing.py for the regression this
    guards against — this is a real trap, not a hypothetical one).
    """
    dyes = np.asarray(dye_reflectances, dtype=np.float64)
    conc = np.asarray(concentrations, dtype=np.float64)

    if dyes.ndim == 1:
        dyes = dyes[np.newaxis, :]

    if np.any(conc < 0):
        raise ValueError("concentrations must be non-negative")
    if conc.shape[0] != dyes.shape[0]:
        raise ValueError(
            f"dye/concentration count mismatch: {dyes.shape[0]} dyes vs "
            f"{conc.shape[0]} concentrations"
        )

    n_channels = dyes.shape[1]
    substrate = (
        np.ones(n_channels, dtype=np.float64)
        if substrate_reflectance is None
        else np.asarray(substrate_reflectance, dtype=np.float64)
    )

    ks_mix = reflectance_to_ks(substrate) + conc @ reflectance_to_ks(dyes)
    return np.clip(ks_to_reflectance(ks_mix), 0.0, 1.0)


def mix_additive_rgb(dye_reflectances, concentrations, substrate_reflectance=None):
    """Concentration-weighted average in raw sRGB. This is the incumbent's
    mixing model, kept only so tests can contrast it against mix_subtractive.

    Models additive light mixing, which is physically wrong for dyes (which
    are subtractive colourants). substrate_reflectance is accepted for
    interface parity but is not physically meaningful to this model and is
    ignored unless concentrations do not sum to 1, in which case it fills the
    remainder.
    """
    dyes = np.asarray(dye_reflectances, dtype=np.float64)
    conc = np.asarray(concentrations, dtype=np.float64)

    if dyes.ndim == 1:
        dyes = dyes[np.newaxis, :]

    if np.any(conc < 0):
        raise ValueError("concentrations must be non-negative")
    if conc.shape[0] != dyes.shape[0]:
        raise ValueError(
            f"dye/concentration count mismatch: {dyes.shape[0]} dyes vs "
            f"{conc.shape[0]} concentrations"
        )

    total = np.sum(conc)
    n_channels = dyes.shape[1]
    substrate = (
        np.ones(n_channels, dtype=np.float64)
        if substrate_reflectance is None
        else np.asarray(substrate_reflectance, dtype=np.float64)
    )

    if total <= 0:
        return substrate.copy()

    weighted = conc @ dyes
    if total < 1.0:
        weighted = weighted + (1.0 - total) * substrate
        total = 1.0

    return weighted / total
