"""In-house transfer-matrix engine for Im(rpp) rows.

This module is the numerical core of the solver. It implements the generalized
4x4 transfer-matrix algorithm (Passler & Paarmann 2017) directly in NumPy,
batched over the in-plane momentum axis: all kx samples of a single
frequency/angle row are processed in one set of array operations. It uses
the :mod:`multilayer_atm.engine` containers.

:func:`compute_row_batched` evaluates a whole row at once. It assumes
* the four out-of-plane modes split cleanly into two forward and two backward
  modes (the physical case for passive media), and
* every 4x4 boundary matrix is invertible.

If a row violates the first assumption :class:`FastPathUnavailable` is raised so
the caller can fall back to :func:`compute_row_pointwise`, which evaluates each
sample independently and maps the rare ill-conditioned point to zero. The two
paths agree to floating-point round-off; the batched path is simply faster by
amortising LAPACK/Python overhead.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import engine
from .materials import CM1_TO_HZ

# Constants and thresholds from the engine so the batched path makes the same
# branch decisions as the (now also engine-based) per-point path.
_C_CONST = engine.C_CONST
_QSD_THR = engine.QSD_THR
_ZERO_THR = engine.ZERO_THR
_ZETA_IMAG_REG = 1e-14

_DELTA1234 = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.complex128,
)


class FastPathUnavailable(RuntimeError):
    """Raised when a row cannot be evaluated by the vectorised path."""


def _batched_exact_inv(matrices: np.ndarray) -> np.ndarray:
    """Analytic inverse of a stack of 4x4 matrices, ``matrices`` shape (N, 4, 4).

    Analytic 4x4 cofactor inverse, batched. Singular matrices yield non-finite
    entries (rather than a pseudo-inverse); the caller treats the resulting
    non-finite rpp as zero.
    """
    a = np.swapaxes(matrices, -1, -2)

    def e(i: int, j: int) -> np.ndarray:
        return a[..., i, j]

    det = (
        e(0, 0) * e(1, 1) * e(2, 2) * e(3, 3) + e(0, 0) * e(1, 2) * e(2, 3) * e(3, 1) + e(0, 0) * e(1, 3) * e(2, 1) * e(3, 2)
        + e(0, 1) * e(1, 0) * e(2, 3) * e(3, 2) + e(0, 1) * e(1, 2) * e(2, 0) * e(3, 3) + e(0, 1) * e(1, 3) * e(2, 2) * e(3, 0)
        + e(0, 2) * e(1, 0) * e(2, 1) * e(3, 3) + e(0, 2) * e(1, 1) * e(2, 3) * e(3, 0) + e(0, 2) * e(1, 3) * e(2, 0) * e(3, 1)
        + e(0, 3) * e(1, 0) * e(2, 2) * e(3, 1) + e(0, 3) * e(1, 1) * e(2, 0) * e(3, 2) + e(0, 3) * e(1, 2) * e(2, 1) * e(3, 0)
        - e(0, 0) * e(1, 1) * e(2, 3) * e(3, 2) - e(0, 0) * e(1, 2) * e(2, 1) * e(3, 3) - e(0, 0) * e(1, 3) * e(2, 2) * e(3, 1)
        - e(0, 1) * e(1, 0) * e(2, 2) * e(3, 3) - e(0, 1) * e(1, 2) * e(2, 3) * e(3, 0) - e(0, 1) * e(1, 3) * e(2, 0) * e(3, 2)
        - e(0, 2) * e(1, 0) * e(2, 3) * e(3, 1) - e(0, 2) * e(1, 1) * e(2, 0) * e(3, 3) - e(0, 2) * e(1, 3) * e(2, 1) * e(3, 0)
        - e(0, 3) * e(1, 0) * e(2, 1) * e(3, 2) - e(0, 3) * e(1, 1) * e(2, 2) * e(3, 0) - e(0, 3) * e(1, 2) * e(2, 0) * e(3, 1)
    )

    b = np.empty_like(a)
    b[..., 0, 0] = e(1, 1) * e(2, 2) * e(3, 3) + e(1, 2) * e(2, 3) * e(3, 1) + e(1, 3) * e(2, 1) * e(3, 2) - e(1, 1) * e(2, 3) * e(3, 2) - e(1, 2) * e(2, 1) * e(3, 3) - e(1, 3) * e(2, 2) * e(3, 1)
    b[..., 0, 1] = e(0, 1) * e(2, 3) * e(3, 2) + e(0, 2) * e(2, 1) * e(3, 3) + e(0, 3) * e(2, 2) * e(3, 1) - e(0, 1) * e(2, 2) * e(3, 3) - e(0, 2) * e(2, 3) * e(3, 1) - e(0, 3) * e(2, 1) * e(3, 2)
    b[..., 0, 2] = e(0, 1) * e(1, 2) * e(3, 3) + e(0, 2) * e(1, 3) * e(3, 1) + e(0, 3) * e(1, 1) * e(3, 2) - e(0, 1) * e(1, 3) * e(3, 2) - e(0, 2) * e(1, 1) * e(3, 3) - e(0, 3) * e(1, 2) * e(3, 1)
    b[..., 0, 3] = e(0, 1) * e(1, 3) * e(2, 2) + e(0, 2) * e(1, 1) * e(2, 3) + e(0, 3) * e(1, 2) * e(2, 1) - e(0, 1) * e(1, 2) * e(2, 3) - e(0, 2) * e(1, 3) * e(2, 1) - e(0, 3) * e(1, 1) * e(2, 2)
    b[..., 1, 0] = e(1, 0) * e(2, 3) * e(3, 2) + e(1, 2) * e(2, 0) * e(3, 3) + e(1, 3) * e(2, 2) * e(3, 0) - e(1, 0) * e(2, 2) * e(3, 3) - e(1, 2) * e(2, 3) * e(3, 0) - e(1, 3) * e(2, 0) * e(3, 2)
    b[..., 1, 1] = e(0, 0) * e(2, 2) * e(3, 3) + e(0, 2) * e(2, 3) * e(3, 0) + e(0, 3) * e(2, 0) * e(3, 2) - e(0, 0) * e(2, 3) * e(3, 2) - e(0, 2) * e(2, 0) * e(3, 3) - e(0, 3) * e(2, 2) * e(3, 0)
    b[..., 1, 2] = e(0, 0) * e(1, 3) * e(3, 2) + e(0, 2) * e(1, 0) * e(3, 3) + e(0, 3) * e(1, 2) * e(3, 0) - e(0, 0) * e(1, 2) * e(3, 3) - e(0, 2) * e(1, 3) * e(3, 0) - e(0, 3) * e(1, 0) * e(3, 2)
    b[..., 1, 3] = e(0, 0) * e(1, 2) * e(2, 3) + e(0, 2) * e(1, 3) * e(2, 0) + e(0, 3) * e(1, 0) * e(2, 2) - e(0, 0) * e(1, 3) * e(2, 2) - e(0, 2) * e(1, 0) * e(2, 3) - e(0, 3) * e(1, 2) * e(2, 0)
    b[..., 2, 0] = e(1, 0) * e(2, 1) * e(3, 3) + e(1, 1) * e(2, 3) * e(3, 0) + e(1, 3) * e(2, 0) * e(3, 1) - e(1, 0) * e(2, 3) * e(3, 1) - e(1, 1) * e(2, 0) * e(3, 3) - e(1, 3) * e(2, 1) * e(3, 0)
    b[..., 2, 1] = e(0, 0) * e(2, 3) * e(3, 1) + e(0, 1) * e(2, 0) * e(3, 3) + e(0, 3) * e(2, 1) * e(3, 0) - e(0, 0) * e(2, 1) * e(3, 3) - e(0, 1) * e(2, 3) * e(3, 0) - e(0, 3) * e(2, 0) * e(3, 1)
    b[..., 2, 2] = e(0, 0) * e(1, 1) * e(3, 3) + e(0, 1) * e(1, 3) * e(3, 0) + e(0, 3) * e(1, 0) * e(3, 1) - e(0, 0) * e(1, 3) * e(3, 1) - e(0, 1) * e(1, 0) * e(3, 3) - e(0, 3) * e(1, 1) * e(3, 0)
    b[..., 2, 3] = e(0, 0) * e(1, 3) * e(2, 1) + e(0, 1) * e(1, 0) * e(2, 3) + e(0, 3) * e(1, 1) * e(2, 0) - e(0, 0) * e(1, 1) * e(2, 3) - e(0, 1) * e(1, 3) * e(2, 0) - e(0, 3) * e(1, 0) * e(2, 1)
    b[..., 3, 0] = e(1, 0) * e(2, 2) * e(3, 1) + e(1, 1) * e(2, 0) * e(3, 2) + e(1, 2) * e(2, 1) * e(3, 0) - e(1, 0) * e(2, 1) * e(3, 2) - e(1, 1) * e(2, 2) * e(3, 0) - e(1, 2) * e(2, 0) * e(3, 1)
    b[..., 3, 1] = e(0, 0) * e(2, 1) * e(3, 2) + e(0, 1) * e(2, 2) * e(3, 0) + e(0, 2) * e(2, 0) * e(3, 1) - e(0, 0) * e(2, 2) * e(3, 1) - e(0, 1) * e(2, 0) * e(3, 2) - e(0, 2) * e(2, 1) * e(3, 0)
    b[..., 3, 2] = e(0, 0) * e(1, 2) * e(3, 1) + e(0, 1) * e(1, 0) * e(3, 2) + e(0, 2) * e(1, 1) * e(3, 0) - e(0, 0) * e(1, 1) * e(3, 2) - e(0, 1) * e(1, 2) * e(3, 0) - e(0, 2) * e(1, 0) * e(3, 1)
    b[..., 3, 3] = e(0, 0) * e(1, 1) * e(2, 2) + e(0, 1) * e(1, 2) * e(2, 0) + e(0, 2) * e(1, 0) * e(2, 1) - e(0, 0) * e(1, 2) * e(2, 1) - e(0, 1) * e(1, 0) * e(2, 2) - e(0, 2) * e(1, 1) * e(2, 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.swapaxes(b, -1, -2) / det[..., None, None]
    return out


def _batched_delta(epsilon: np.ndarray, mu: complex, z: np.ndarray) -> np.ndarray:
    """Build the (N, 4, 4) Delta matrices for a layer over the zeta array ``z``."""
    # Constitutive matrix entries actually referenced by calculate_matrices. Only
    # the upper-left (epsilon) and lower-right (mu) blocks are non-zero; the
    # magneto-electric blocks vanish, so the corresponding M terms are dropped.
    M22 = epsilon[2, 2]
    M55 = mu
    b = M22 * M55

    M20, M21 = epsilon[2, 0], epsilon[2, 1]
    M00, M01, M02 = epsilon[0, 0], epsilon[0, 1], epsilon[0, 2]
    M10, M11, M12 = epsilon[1, 0], epsilon[1, 1], epsilon[1, 2]
    M33 = M44 = mu

    # a-matrix (eqn 9). Entries that are identically zero for non-bianisotropic
    # media are omitted; the constitutive matrix has M[2,5]=M[5,2]=0.
    a20 = (-M20 * M55) / b
    a21 = (-M21 * M55) / b
    a24 = (-(z) * M55) / b
    a51 = (M22 * z) / b

    # S-matrix (the four columns used by Delta).
    S00 = M00 + M02 * a20
    S01 = M01 + M02 * a21
    S03 = M02 * a24
    S10 = M10 + M12 * a20
    S11 = M11 + M12 * a21 + (-z) * a51
    S13 = M12 * a24
    S22 = M33
    S30 = z * a20
    S31 = z * a21
    S33 = M44 + z * a24

    n = z.shape[0]
    delta = np.zeros((n, 4, 4), dtype=np.complex128)
    delta[:, 0, 0] = S30
    delta[:, 0, 1] = S33
    delta[:, 0, 2] = S31
    delta[:, 0, 3] = 0.0  # -S32, S32 == 0
    delta[:, 1, 0] = S00
    delta[:, 1, 1] = S03
    delta[:, 1, 2] = S01
    delta[:, 1, 3] = 0.0  # -S02, S02 == 0
    delta[:, 2, 0] = 0.0  # -S20
    delta[:, 2, 1] = 0.0  # -S23
    delta[:, 2, 2] = 0.0  # -S21
    delta[:, 2, 3] = S22
    delta[:, 3, 0] = S10
    delta[:, 3, 1] = S13
    delta[:, 3, 2] = S11
    delta[:, 3, 3] = 0.0  # -S12
    return delta, (a20, a21, a24, a51)


def _clean_small(values: np.ndarray) -> np.ndarray:
    """Zero out real/imaginary parts below ``zero_thr`` (mirrors calculate_q)."""
    out = values.copy()
    im = out.imag
    re = out.real
    mask_im = (np.abs(im) > 0) & (np.abs(im) < _ZERO_THR)
    out = np.where(mask_im, out.real + 0.0j, out)
    re = out.real
    im = out.imag
    mask_re = (np.abs(re) > 0) & (np.abs(re) < _ZERO_THR)
    out = np.where(mask_re, 0.0 + 1.0j * im, out)
    return out


def _isotropic_forward_q(D: np.ndarray) -> np.ndarray:
    """Forward (transmitted) root ``q = sqrt(D)`` with the engine's branch convention."""
    r = np.sqrt(D)
    flip = (np.abs(r.imag) >= _ZERO_THR) & (r.imag < 0.0)
    return np.where(flip, -r, r)


def _isotropic_qs_gamma(eps: complex, mu: complex, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form qs/gamma for an isotropic layer (eps scalar).

    For a scalar tensor the Berreman matrix is block-diagonal and all four modes
    are q = +/- sqrt(eps*mu - zeta^2), forward modes doubly degenerate. This
    reproduces, in closed form, exactly what the eigen path produces in its
    degenerate non-birefringent branch (p-mode at index 0/2, s-mode at index 1/3),
    so it is numerically identical while skipping np.linalg.eig and the mode sort.
    """
    n = z.shape[0]
    D = mu * eps - z ** 2
    # The forward (transmitted) root must match the convention the eigen path's mode
    # sort uses: Im(q) >= 0 for evanescent samples, Re(q) > 0 for propagating ones.
    # np.sqrt is the principal root (Re >= 0), which already satisfies Re > 0 for
    # propagating samples; only the evanescent branch (D on the negative real axis,
    # where the principal root is ~ -i|q|) needs its sign flipped to get Im >= 0.
    qt = _isotropic_forward_q(D)

    qs = np.empty((n, 4), dtype=np.complex128)
    qs[:, 0] = qt
    qs[:, 1] = qt
    qs[:, 2] = -qt
    qs[:, 3] = -qt

    with np.errstate(divide="ignore", invalid="ignore"):
        g_p = -(z * qt) / D  # gamma13 (forward p) == gamma33 (backward p) here

    ones = np.ones(n, dtype=np.complex128)
    zeros = np.zeros(n, dtype=np.complex128)
    gamma = np.stack(
        [
            np.stack([ones, zeros, g_p], axis=-1),    # forward p
            np.stack([zeros, ones, zeros], axis=-1),  # forward s
            np.stack([-ones, zeros, g_p], axis=-1),   # backward p
            np.stack([zeros, ones, zeros], axis=-1),  # backward s
        ],
        axis=1,
    )
    norms = np.linalg.norm(gamma, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = gamma / norms
    return qs, gamma


def _batched_qs_gamma(layer: engine.Layer, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted out-of-plane wavevectors qs (N, 4) and field matrix gamma (N, 4, 3)."""
    epsilon = layer.epsilon
    mu = layer.mu
    n = z.shape[0]

    # Isotropic layers (vacuum/air, cubic crystals, metals, ...) have analytic
    # modes; skip the eigen-decomposition and mode-sorting entirely.
    if getattr(layer, "is_isotropic", False):
        return _isotropic_qs_gamma(epsilon[0, 0], mu, z)

    delta, (a20, a21, a24, a51) = _batched_delta(epsilon, mu, z)

    # Eigen-decomposition of every Delta at once.
    qs_un, psi_un = np.linalg.eig(delta)  # qs_un (N,4), psi_un (N,4,4) cols = vecs
    qs_un = _clean_small(qs_un)

    # Poynting vectors and Berreman fields for each unsorted mode (eqns 16-18).
    Ex = psi_un[:, 0, :]
    Ey = psi_un[:, 2, :]
    Hx = -psi_un[:, 3, :]
    Hy = psi_un[:, 1, :]
    a20c = np.asarray(a20).reshape(-1, 1) if np.ndim(a20) else a20
    a21c = np.asarray(a21).reshape(-1, 1) if np.ndim(a21) else a21
    a24c = a24.reshape(-1, 1)
    a51c = a51.reshape(-1, 1)
    Ez = a20c * Ex + a21c * Ey + a24c * Hy  # a23 == 0
    Hz = a51c * Ey  # a50 = a53 = a54 == 0
    Py0 = Ey * Hz - Ez * Hy
    Py1 = Ez * Hx - Ex * Hz
    berreman = np.stack([Ex, Ey, Ez], axis=-1)  # (N, 4, 3)

    # Partition into transmitted (forward) and reflected (backward) modes, keeping
    # ascending eigen-index order within each group exactly as the reference loop.
    has_imag = np.any(np.abs(qs_un.imag) > 0, axis=1, keepdims=True)
    cond = np.where(has_imag, qs_un.imag >= 0, qs_un.real > 0)
    if not np.all(cond.sum(axis=1) == 2):
        raise FastPathUnavailable("modes do not split 2/2 into forward/backward")

    km = np.arange(4)
    key = np.where(cond, km, km + 100)
    order = np.argsort(key, axis=1, kind="stable")
    trans = order[:, :2]
    refl = order[:, 2:]

    def gather1(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return np.take_along_axis(arr, idx, axis=1)

    Py0_t, Py1_t = gather1(Py0, trans), gather1(Py1, trans)
    Py0_r, Py1_r = gather1(Py0, refl), gather1(Py1, refl)
    psi0 = psi_un[:, 0, :]
    psi2 = psi_un[:, 2, :]
    psi0_t, psi2_t = gather1(psi0, trans), gather1(psi2, trans)
    psi0_r, psi2_r = gather1(psi0, refl), gather1(psi2, refl)

    with np.errstate(divide="ignore", invalid="ignore"):
        Cp_t1 = np.abs(Py0_t[:, 0]) ** 2 / (np.abs(Py0_t[:, 0]) ** 2 + np.abs(Py1_t[:, 0]) ** 2)
        Cp_t2 = np.abs(Py0_t[:, 1]) ** 2 / (np.abs(Py0_t[:, 1]) ** 2 + np.abs(Py1_t[:, 1]) ** 2)
        biref = np.abs(Cp_t1 - Cp_t2) > _QSD_THR

        # Birefringent ranking uses Poynting vectors; otherwise the E-field.
        Cp_r1 = np.abs(Py0_r[:, 1]) ** 2 / (np.abs(Py0_r[:, 1]) ** 2 + np.abs(Py1_r[:, 1]) ** 2)
        Cp_r2 = np.abs(Py0_r[:, 0]) ** 2 / (np.abs(Py0_r[:, 0]) ** 2 + np.abs(Py1_r[:, 0]) ** 2)
        Cp_te1 = np.abs(psi0_t[:, 1]) ** 2 / (np.abs(psi0_t[:, 1]) ** 2 + np.abs(psi2_t[:, 1]) ** 2)
        Cp_te2 = np.abs(psi0_t[:, 0]) ** 2 / (np.abs(psi0_t[:, 0]) ** 2 + np.abs(psi2_t[:, 0]) ** 2)
        Cp_re1 = np.abs(psi0_r[:, 1]) ** 2 / (np.abs(psi0_r[:, 1]) ** 2 + np.abs(psi2_r[:, 1]) ** 2)
        Cp_re2 = np.abs(psi0_r[:, 0]) ** 2 / (np.abs(psi0_r[:, 0]) ** 2 + np.abs(psi2_r[:, 0]) ** 2)

    flip_trans = np.where(biref, Cp_t2 > Cp_t1, Cp_te1 > Cp_te2)
    flip_refl = np.where(biref, Cp_r1 > Cp_r2, Cp_re1 > Cp_re2)
    trans = np.where(flip_trans[:, None], trans[:, ::-1], trans)
    refl = np.where(flip_refl[:, None], refl[:, ::-1], refl)

    full = np.concatenate([trans, refl], axis=1)  # (N, 4): [t-p, t-s, r-p, r-s]
    qs = np.take_along_axis(qs_un, full, axis=1)
    berreman_sorted = np.take_along_axis(berreman, full[:, :, None] + np.zeros((1, 1, 3), dtype=int), axis=1)

    gamma = _batched_gamma(epsilon, mu, z, qs, berreman_sorted, biref)
    return qs, gamma


def _batched_gamma(
    epsilon: np.ndarray,
    mu: complex,
    z: np.ndarray,
    qs: np.ndarray,
    berreman_sorted: np.ndarray,
    biref: np.ndarray,
) -> np.ndarray:
    """Field-vector matrix gamma (N, 4, 3), Xu fields with Berreman substitution."""
    e00, e01, e02 = epsilon[0, 0], epsilon[0, 1], epsilon[0, 2]
    e10, e11 = epsilon[1, 0], epsilon[1, 1]
    e12 = epsilon[1, 2]
    e20, e21, e22 = epsilon[2, 0], epsilon[2, 1], epsilon[2, 2]

    q0, q1, q2, q3 = qs[:, 0], qs[:, 1], qs[:, 2], qs[:, 3]
    D = mu * e22 - z ** 2

    def nan_to(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(value), fallback, value)

    with np.errstate(divide="ignore", invalid="ignore"):
        # --- transmitted pair (modes 0, 1) ---
        deg01 = np.abs(q0 - q1) < _QSD_THR

        g12_full = nan_to(
            (mu * e12 * (mu * e20 + z * q0) - mu * e10 * D)
            / (D * (mu * e11 - z ** 2 - q0 ** 2) - mu ** 2 * e12 * e21),
            np.zeros_like(z),
        )
        g13_full = nan_to(
            (-(mu * e20 + z * q0) - mu * e21 * g12_full) / D,
            (-(mu * e20 + z * q0)) / D,
        )
        g21_full = nan_to(
            (mu * e21 * (mu * e02 + z * q1) - mu * e01 * D)
            / (D * (mu * e00 - q1 ** 2) - (mu * e02 + z * q1) * (mu * e20 + z * q1)),
            np.zeros_like(z),
        )
        g23_full = nan_to(
            (-(mu * e20 + z * q1) * g21_full - mu * e21) / D,
            (-mu * e21) / D,
        )

        g12 = np.where(deg01, 0.0, g12_full)
        g13 = np.where(deg01, -(mu * e20 + z * q0) / D, g13_full)
        g21 = np.where(deg01, 0.0, g21_full)
        g23 = np.where(deg01, -mu * e21 / D, g23_full)

        # --- reflected pair (modes 2, 3) ---
        deg23 = np.abs(q2 - q3) < _QSD_THR

        g32_full = nan_to(
            (mu * e10 * D - mu * e12 * (mu * e20 + z * q2))
            / (D * (mu * e11 - z ** 2 - q2 ** 2) - mu ** 2 * e12 * e21),
            np.zeros_like(z),
        )
        g33_full = nan_to(
            ((mu * e20 + z * q2) + mu * e21 * g32_full) / D,
            (mu * e20 + z * q2) / D,
        )
        g41_full = nan_to(
            (mu * e21 * (mu * e02 + z * q3) - mu * e01 * D)
            / (D * (mu * e00 - q3 ** 2) - (mu * e02 + z * q3) * (mu * e20 + z * q3)),
            np.zeros_like(z),
        )
        g43_full = nan_to(
            (-(mu * e20 + z * q3) * g41_full - mu * e21) / D,
            (-mu * e21) / D,
        )

        g32 = np.where(deg23, 0.0, g32_full)
        g33 = np.where(deg23, (mu * e20 + z * q2) / D, g33_full)
        g41 = np.where(deg23, 0.0, g41_full)
        g43 = np.where(deg23, -mu * e21 / D, g43_full)

    n = z.shape[0]
    ones = np.ones(n, dtype=np.complex128)
    g1 = np.stack([ones, g12, g13], axis=-1)
    g2 = np.stack([g21, ones, g23], axis=-1)
    g3 = np.stack([-ones, g32, g33], axis=-1)
    g4 = np.stack([g41, ones, g43], axis=-1)
    gamma_xu = np.stack([g1, g2, g3, g4], axis=1)  # (N, 4, 3)

    norms = np.linalg.norm(gamma_xu, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_xu = gamma_xu / norms

    b_norms = np.linalg.norm(berreman_sorted, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_b = berreman_sorted / b_norms

    return np.where(biref[:, None, None], gamma_b, gamma_xu)


def _batched_ai(layer: engine.Layer, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qs, gamma = _batched_qs_gamma(layer, z)
    mu = layer.mu
    n = z.shape[0]
    ai = np.empty((n, 4, 4), dtype=np.complex128)
    ai[:, 0, :] = gamma[:, :, 0]
    ai[:, 1, :] = gamma[:, :, 1]
    ai[:, 2, :] = (qs * gamma[:, :, 0] - z[:, None] * gamma[:, :, 2]) / mu
    ai[:, 3, :] = qs * gamma[:, :, 1] / mu
    return ai, qs


def _batched_ti(layer: engine.Layer, f_hz: float, z: np.ndarray) -> np.ndarray:
    ai, qs = _batched_ai(layer, z)
    phase = (-2.0j * np.pi * f_hz / _C_CONST) * (qs * layer.thick)
    n = z.shape[0]
    ki = np.zeros((n, 4, 4), dtype=np.complex128)
    idx = np.arange(4)
    ki[:, idx, idx] = np.exp(phase)
    ai_inv = _batched_exact_inv(ai)
    return ai @ (ki @ ai_inv)


def _batched_gamma_star(system: engine.System, f_hz: float, z: np.ndarray) -> np.ndarray:
    ai_super, _ = _batched_ai(system.superstrate, z)
    ai_inv_super = _batched_exact_inv(ai_super)
    ai_sub, _ = _batched_ai(system.substrate, z)

    n = z.shape[0]
    tloc = np.broadcast_to(np.identity(4, dtype=np.complex128), (n, 4, 4)).copy()
    for ii in range(len(system.layers) - 1, -1, -1):
        tloc = _batched_ti(system.layers[ii], f_hz, z) @ tloc

    gamma = ai_inv_super @ (tloc @ ai_sub)
    gamma_star = _DELTA1234 @ gamma @ _DELTA1234
    return gamma_star


# ---------------------------------------------------------------------------
# Fast path: propagate only the 4x2 block that rpp depends on
# ---------------------------------------------------------------------------
#
# Maths used to simplify the per-layer work (all exact, no approximation):
#
# 1. The layer transfer matrix T_i = A_i K_i A_i^-1 equals exp(-i*alpha*Delta_i)
#    (alpha = 2*pi*f*d/c) up to the basis permutation (Ex,Ey,Hy,-Hx) <-> Berreman
#    (Ex,Hy,Ey,-Hx). K_i is diagonal, and any rescaling / re-ordering /
#    re-basis *inside* a degenerate eigenspace of Delta_i cancels in A K A^-1.
#    The field-vector construction (Poynting-vector mode ranking, Xu/Berreman
#    gamma, forward/backward split, ``take_along_axis`` gathers) is therefore
#    irrelevant for interior layers: T_i = Psi K Psi^-1 with Psi the raw
#    ``np.linalg.eig`` eigenvectors of Delta_i. Only the semi-infinite
#    boundaries need sorted modes.
#
# 2. For an isotropic layer Delta^2 = q^2 * 1 (both the p and the s 2x2 block
#    square to q^2 = eps*mu - zeta^2), so by Cayley-Hamilton
#        exp(-i*alpha*Delta) = cos(alpha*q) * 1 - i*sin(alpha*q)/q * Delta.
#    This is even in q (no branch choice) and needs no eig and no inverse.
#
# 3. rpp = numer/denom only involves gamma_star[:3, columns 0 and 2], i.e. the
#    first two (transmitted) columns of A_substrate and three rows of
#    A_superstrate^-1. Instead of accumulating the full 4x4 product we push the
#    (N,4,2) block A_sub[:, :, :2] through the layers (a 4x4 @ 4x2 product, half
#    the flops) and never form the full T_total. For anisotropic layers
#    T M = Psi K (Psi^-1 M) is evaluated with one batched ``solve`` (2 right-hand
#    sides) instead of an explicit 4x4 inverse followed by two 4x4 products.

_PERM = np.array([0, 2, 1, 3])  # Passler (Ex,Ey,Hy,-Hx) <-> Berreman (Ex,Hy,Ey,-Hx)


def _propagate_layer(
    layer: engine.Layer, f_hz: float, z: np.ndarray, m: np.ndarray, force_eig: bool = False
) -> np.ndarray:
    """Return ``T_layer @ m`` for ``m`` of shape (N, 4, k), in the Berreman basis.

    ``force_eig`` disables the closed forms (isotropic / block-diagonal) and always
    uses the generic eig path; it is the reference the closed forms are tested against.
    """
    alpha = (2.0 * np.pi * f_hz / _C_CONST) * layer.thick
    eps = layer.epsilon
    mu = layer.mu

    if not force_eig:
        if getattr(layer, "is_isotropic", False):
            return _propagate_isotropic(eps[0, 0], mu, alpha, z, m)
        if getattr(layer, "is_block_diagonal", False):
            return _propagate_block_diagonal(eps, mu, alpha, z, m)
    return _propagate_eig(eps, mu, alpha, z, m)


# Below this growth X = alpha*|Im q| the cos/sin form of an isotropic layer is as accurate
# as any (the cancelled sub-dominant part is only e^{-2X} ~ 2% of the total); above it the
# diagonalised form is used. The diagonalised form divides by q, which costs digits
# (~eps/|q|) exactly where cos/sin is perfectly stable: near the light line of the layer.
_ISO_DIAG_GROWTH = 2.0


def _propagate_isotropic(e: complex, mu: complex, alpha, z: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Isotropic layer applied to ``m`` without LAPACK.

    Delta is two decoupled 2x2 blocks, p on rows (Ex, Hy) and s on rows (Ey, -Hx), each
    ``[[0, b], [a, 0]]`` with ``a*b = q^2 = eps*mu - zeta^2`` (p: ``b = mu - z^2/eps,
    a = eps``; s: ``b = mu, a = eps - z^2/mu``), so ``Delta^2 = q^2`` and

        exp(-i alpha Delta) = cos(alpha q) 1 - i sin(alpha q)/q Delta          (A)

    (even in q: no branch choice; ``sin(x)/x`` is evaluated safely at q = 0). For a thick
    or strongly evanescent layer (A) adds two e^{+X} terms (X = alpha |Im q|) that must
    cancel to leave the e^{-X} part, so there (X > ``_ISO_DIAG_GROWTH``) the algebraically
    identical diagonalised form is used: the eigenvectors of each block are
    ``(1, +-a/q)`` for the eigenvalues ``+-q``, with mode amplitudes
    ``c+- = (u0 +- (q/a) u1) / 2`` and

        T u = (E+ c+ + E- c-,  (a/q) (E+ c+ - E- c-)),   E+- = exp(-+ i alpha q),

    which keeps the growing and decaying exponentials separate (the accuracy of the
    eig-based factorisation).
    """
    q = np.sqrt(mu * e - z ** 2)
    aq = alpha * q
    with np.errstate(all="ignore"):
        sinc_q = np.where(np.abs(q) < 1e-30, alpha, np.sin(aq) / q)
    c = np.cos(aq)[:, None]
    s = (-1j * sinc_q)[:, None]
    d01 = (mu - z ** 2 / e)[:, None]   # Delta[Ex, Hy]
    d32 = (e - z ** 2 / mu)[:, None]   # Delta[-Hx, Ey]
    e_col = np.asarray(e).reshape(-1, 1) if np.ndim(e) else e
    out = np.empty_like(m)
    out[:, 0] = c * m[:, 0] + (s * d01) * m[:, 1]
    out[:, 1] = c * m[:, 1] + (s * e_col) * m[:, 0]
    out[:, 2] = c * m[:, 2] + (s * mu) * m[:, 3]
    out[:, 3] = c * m[:, 3] + (s * d32) * m[:, 2]

    with np.errstate(invalid="ignore"):
        grown = np.abs(np.asarray(alpha) * q.imag) > _ISO_DIAG_GROWTH
    if grown.any():
        e_g = e[grown] if np.ndim(e) else e
        al = alpha[grown] if np.ndim(alpha) else alpha
        qg, zg, mg = q[grown], z[grown], m[grown]
        # Common per-sample factor e^{-X} keeps the larger exponential at 1 (it would
        # overflow beyond X ~ 709); it scales both columns alike, so rpp is unchanged.
        shift = np.abs(al * qg.imag)
        ep = np.exp(-1j * al * qg - shift)[:, None]
        em = np.exp(1j * al * qg - shift)[:, None]
        for (i0, i1), a in (((0, 1), np.broadcast_to(e_g, qg.shape)), ((2, 3), e_g - zg ** 2 / mu)):
            with np.errstate(all="ignore"):
                r = (qg / a)[:, None]
                cp = 0.5 * (mg[:, i0] + r * mg[:, i1])
                cm = 0.5 * (mg[:, i0] - r * mg[:, i1])
                sub = out[grown]
                sub[:, i0] = ep * cp + em * cm
                sub[:, i1] = (ep * cp - em * cm) / r
                out[grown] = sub
    return out


def _take(eps: np.ndarray, alpha, index: np.ndarray):
    """Subset ``eps`` (3,3 or 3,3,N) and ``alpha`` (scalar or (N,)) to the samples ``index``."""
    return (eps[:, :, index] if np.ndim(eps) == 3 else eps), (alpha[index] if np.ndim(alpha) else alpha)


def _propagate_eig(eps: np.ndarray, mu: complex, alpha, z: np.ndarray, m: np.ndarray) -> np.ndarray:
    """General anisotropic layer: ``T m = Psi K (Psi^-1 m)`` from a batched eig.

    Samples with a non-finite Delta (overflow) or a singular eigenvector matrix
    yield NaN for those samples only, instead of failing the whole batch.
    """
    delta, _ = _batched_delta(eps, mu, z)
    finite = np.isfinite(delta).all(axis=(1, 2))
    if not finite.all():
        out = np.full_like(m, np.nan)
        if finite.any():
            eps_f, alpha_f = _take(eps, alpha, finite)
            out[finite] = _propagate_eig(eps_f, mu, alpha_f, z[finite], m[finite])
        return out
    qs, psi = np.linalg.eig(delta)
    try:
        y = np.linalg.solve(psi, m)
    except np.linalg.LinAlgError:
        out = np.full_like(m, np.nan)
        for i in range(z.shape[0]):
            try:
                a_i = alpha[i] if np.ndim(alpha) else alpha
                yi = np.linalg.solve(psi[i], m[i])
                out[i] = psi[i] @ (np.exp(-1j * a_i * qs[i])[:, None] * yi)
            except np.linalg.LinAlgError:
                pass
        return out
    a_col = np.asarray(alpha)[..., None] if np.ndim(alpha) else alpha
    phase = a_col * qs
    # Thick / strongly evanescent layer: the two growing partial waves have different
    # rates, so the summed columns of ``psi @ (E y)`` become numerically collinear
    # (the slower mode drowns in rounding noise) and rpp turns to garbage. rpp only
    # depends on the column *span* of m, so replace the columns by the equivalent
    # ``psi_g + psi_d (E_d/E_g) (y_d y_g^-1)``: same span up to the decaying part
    # (relative size E_d/E_g), with no large exponentials formed.
    log_e = phase.imag  # log|exp(-i*phase)|
    wide = (log_e.max(axis=1) - log_e.min(axis=1)) > _EIG_SPREAD_STABILISE
    if not wide.any():
        y *= np.exp(-1j * phase)[:, :, None]
        return psi @ y
    out = np.empty_like(m)
    narrow = ~wide
    if narrow.any():
        yn = y[narrow] * np.exp(-1j * phase[narrow])[:, :, None]
        out[narrow] = psi[narrow] @ yn
    order = np.argsort(-log_e[wide], axis=1)
    g_idx, d_idx = order[:, :2], order[:, 2:]
    pw, yw, qw = psi[wide], y[wide], qs[wide]
    aw = a_col[wide] if np.ndim(alpha) else alpha
    psi_g = np.take_along_axis(pw, g_idx[:, None, :], axis=2)
    psi_d = np.take_along_axis(pw, d_idx[:, None, :], axis=2)
    y_g = np.take_along_axis(yw, g_idx[:, :, None], axis=1)
    y_d = np.take_along_axis(yw, d_idx[:, :, None], axis=1)
    q_g = np.take_along_axis(qw, g_idx, axis=1)
    q_d = np.take_along_axis(qw, d_idx, axis=1)
    try:
        ratio = np.linalg.solve(y_g.transpose(0, 2, 1), y_d.transpose(0, 2, 1)).transpose(0, 2, 1)
    except np.linalg.LinAlgError:
        ratio = np.full((y_g.shape[0], 2, 2), np.nan, dtype=np.complex128)
    aw3 = np.asarray(aw).reshape(-1, 1, 1) if np.ndim(aw) else aw
    decay = np.exp(-1j * aw3 * (q_d[:, :, None] - q_g[:, None, :]))
    out[wide] = psi_g + psi_d @ (ratio * decay)
    return out


# Relative eigenvalue gap below which the closed-form 2x2 interpolation loses
# accuracy (divided differences ~ eps / gap); such samples use the eig path.
_BLOCK_GAP_TOL = 1e-5
# The block-diagonal Cayley-Hamilton form sums terms of size e^{X},
# X = alpha*max|Im sqrt(x_j)| (growth of the layer propagator), and loses
# sub-dominant information that the factorised ``Psi K Psi^-1`` form keeps. Checked
# against a 60-digit matrix exponential the two agree to round-off up to X ~ 10;
# beyond that (very thick / very evanescent layers) those samples use the eig path.
_BLOCK_MAX_GROWTH = 10.0
# Spread of log|exp(-i alpha q)| across a layer's four modes above which the eig path
# switches to the span-preserving factorised form (see _propagate_eig).
_EIG_SPREAD_STABILISE = 12.0


def _propagate_block_diagonal(
    eps: np.ndarray, mu: complex, alpha: float, z: np.ndarray, m: np.ndarray
) -> np.ndarray:
    """Layer with eps_xz = eps_yz = 0: closed-form 2x2 propagation, no eig.

    With ``E_t = (Ex, Ey)`` (rows 0, 2) and ``h = (Hy, -Hx)`` (rows 1, 3) the
    Berreman matrix is ``Delta = [[0, A], [B, 0]]`` with the 2x2 blocks
    ``A = diag(mu - z^2/eps_zz, mu)`` and
    ``B = [[eps_xx, eps_xy], [eps_yx, eps_yy - z^2/mu]]``. Then
    ``Delta^(2k) = diag((AB)^k, (BA)^k)`` and
    ``exp(-i a Delta) = [[C(AB), -i A S(BA)], [-i B S(AB), C(BA)]]`` with
    ``C(x) = cos(a sqrt x)`` and ``S(x) = sin(a sqrt x)/sqrt x`` (both even in
    sqrt x, so no branch choice). AB and BA share the two eigenvalues
    ``x1, x2`` (roots of a 2x2 characteristic polynomial); each 2x2 matrix
    function follows from Cayley-Hamilton: ``f(N) = f0 + f1 N`` with
    ``f1 = (f(x1) - f(x2)) / (x1 - x2)``, ``f0 = f(x1) - f1 x1``.
    """
    exx, exy, eyx, eyy, ezz = eps[0, 0], eps[0, 1], eps[1, 0], eps[1, 1], eps[2, 2]
    a0 = mu - z ** 2 / ezz            # A = diag(a0, mu)
    b11 = eyy - z ** 2 / mu           # B = [[exx, exy], [eyx, b11]]

    # N = A B, P = B A  (2x2, entries as (N,) arrays)
    n00, n01, n10, n11 = a0 * exx, a0 * exy, mu * eyx, mu * b11
    p00, p01, p10, p11 = exx * a0, exy * mu, eyx * a0, b11 * mu

    half_tr = 0.5 * (n00 + n11)
    disc = np.sqrt(half_tr ** 2 - (n00 * n11 - n01 * n10))
    x1, x2 = half_tr + disc, half_tr - disc
    gap = x1 - x2

    r1, r2 = np.sqrt(x1), np.sqrt(x2)
    with np.errstate(all="ignore"):
        c1v, c2v = np.cos(alpha * r1), np.cos(alpha * r2)
        s1v, s2v = np.sin(alpha * r1) / r1, np.sin(alpha * r2) / r2
        c1 = (c1v - c2v) / gap
        s1 = (s1v - s2v) / gap
    c0 = c1v - c1 * x1
    s0 = s1v - s1 * x1

    et0, et1 = m[:, 0], m[:, 2]       # (N, k)
    h0, h1 = m[:, 1], m[:, 3]
    col = lambda v: np.asarray(v).reshape(-1, 1)  # scalar or (N,) -> broadcastable column
    c0, c1, s0, s1 = col(c0), col(c1), col(s0), col(s1)

    def apply(q00, q01, q10, q11, f0, f1, v0, v1):
        # (f0 + f1 Q) v
        return (
            f0 * v0 + f1 * (col(q00) * v0 + col(q01) * v1),
            f0 * v1 + f1 * (col(q10) * v0 + col(q11) * v1),
        )

    ah0, ah1 = col(a0) * h0, mu * h1          # A h
    beh0 = col(exx) * et0 + col(exy) * et1    # B Et
    beh1 = col(eyx) * et0 + col(b11) * et1

    ce0, ce1 = apply(n00, n01, n10, n11, c0, c1, et0, et1)       # C(AB) Et
    sa0, sa1 = apply(n00, n01, n10, n11, s0, s1, ah0, ah1)       # S(AB) A h  (= A S(BA) h)
    sb0, sb1 = apply(p00, p01, p10, p11, s0, s1, beh0, beh1)     # S(BA) B Et (= B S(AB) Et)
    ch0, ch1 = apply(p00, p01, p10, p11, c0, c1, h0, h1)         # C(BA) h

    out = np.empty_like(m)
    out[:, 0] = ce0 - 1j * sa0
    out[:, 2] = ce1 - 1j * sa1
    out[:, 1] = ch0 - 1j * sb0
    out[:, 3] = ch1 - 1j * sb1

    # Near-degenerate eigenvalues (e.g. uniaxial near normal incidence), strongly
    # growing propagators, or a non-finite result: recompute those samples with
    # the general eig path.
    with np.errstate(all="ignore"):
        growth = alpha * np.maximum(np.abs(r1.imag), np.abs(r2.imag))
        bad = (
            ~(np.abs(gap) > _BLOCK_GAP_TOL * (np.abs(x1) + np.abs(x2)))
            | ~np.isfinite(gap)
            | ~(growth <= _BLOCK_MAX_GROWTH)
        )
    if np.any(bad):
        eps_b, alpha_b = _take(eps, alpha, bad)
        out[bad] = _propagate_eig(eps_b, mu, alpha_b, z[bad], m[bad])
    return out


def _normalise_columns(m: np.ndarray) -> np.ndarray:
    """Scale each column of ``m`` (N,4,k) to unit max-norm.

    rpp is a ratio of 2x2 minors of the two columns, so it is invariant under a
    per-column scale; without this a stack of thick evanescent layers overflows
    (e^{X} per layer, squared in the minors) and the sample is wrongly zeroed.
    Non-finite or all-zero columns are left untouched.
    """
    with np.errstate(all="ignore"):
        scale = np.abs(m).max(axis=1, keepdims=True)
        ok = np.isfinite(scale) & (scale > 0)
        return m / np.where(ok, scale, 1.0)


def _rpp_numer_denom(
    system: engine.System, f_hz: float, z: np.ndarray, force_eig: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(numer, denom)`` with ``rpp = numer/denom`` for the whole zeta array.

    ``numer`` and ``denom`` are only defined up to a common (mode-normalisation)
    factor, which cancels in the ratio.
    """
    sub = system.substrate
    if getattr(sub, "is_isotropic", False):
        # Forward p/s modes of an isotropic half-space, Passler rows (Ex, Ey, Hy, -Hx):
        #   p: (1, 0, eps/q, 0)    s: (0, 1, 0, q/mu)
        # (mode normalisation is irrelevant: it scales numer and denom alike).
        e_s, mu_s = sub.epsilon[0, 0], sub.mu
        q_s = _isotropic_forward_q(mu_s * e_s - z ** 2)
        m = np.zeros((z.shape[0], 4, 2), dtype=np.complex128)
        with np.errstate(all="ignore"):
            m[:, 0, 0] = 1.0
            m[:, 2, 0] = e_s / q_s
            m[:, 1, 1] = 1.0
            m[:, 3, 1] = q_s / mu_s
        m = m[:, _PERM, :]  # Berreman basis
    else:
        ai_sub, _ = _batched_ai(sub, z)
        # Transmitted (forward) columns of the substrate, Passler -> Berreman basis.
        m = ai_sub[:, :, :2][:, _PERM, :]

    for layer in reversed(system.layers):
        m = _normalise_columns(_propagate_layer(layer, f_hz, z, m, force_eig=force_eig))
    m = m[:, _PERM, :]  # back to the Passler basis

    sup = system.superstrate
    if getattr(sup, "is_isotropic", False):
        # Rows 0..2 of A_super^-1 (modes: fwd p, fwd s, back p) of an isotropic
        # half-space, with the p rows' common factor 1/(2n) and the s row's 1/2
        # dropped: numer and denom are bilinear in (p row, s row), so these
        # constants cancel in rpp = numer/denom.
        #   fwd p : (1, 0, 1/k, 0)   back p : (-1, 0, 1/k, 0)   with 1/k = q/eps
        #   fwd s : (0, 1, 0, mu/q)
        e_p, mu_p = sup.epsilon[0, 0], sup.mu
        q_p = _isotropic_forward_q(mu_p * e_p - z ** 2)
        with np.errstate(all="ignore"):
            inv_k = (q_p / e_p)[:, None]
            g0 = m[:, 0] + inv_k * m[:, 2]
            g2 = -m[:, 0] + inv_k * m[:, 2]
            g1 = m[:, 1] + (mu_p / q_p)[:, None] * m[:, 3]
    else:
        ai_super, _ = _batched_ai(sup, z)
        g = _batched_exact_inv(ai_super)[:, :3, :] @ m
        g0, g1, g2 = g[:, 0], g[:, 1], g[:, 2]

    denom = g0[:, 0] * g1[:, 1] - g0[:, 1] * g1[:, 0]
    numer = g2[:, 0] * g1[:, 1] - g2[:, 1] * g1[:, 0]
    return numer, denom


def _zeta_from_kx(w_cm1: float, kx_cm1: np.ndarray) -> np.ndarray:
    """Reduced wavevector zeta for a row, lifted off the singular real axis."""
    kx = np.asarray(kx_cm1, dtype=float)
    z = (kx * (1.0 / float(w_cm1))).astype(np.complex128)
    # Input momentum is purely real here, so add the same tiny imaginary
    # regulariser the original per-point evaluation used to avoid the mode-sorting
    # singular branch.
    return z + 1j * _ZETA_IMAG_REG


def _rpp_from_zeta(system: engine.System, f_hz: float, z: np.ndarray) -> np.ndarray:
    """Im(rpp) inputs: rpp values over a zeta array (epsilon must be initialised)."""
    with np.errstate(all="ignore"):
        numer, denom = _rpp_numer_denom(system, f_hz, z)
        out = numer / denom
    return np.where(np.isfinite(out), out, 0.0 + 0.0j)


def compute_row_batched(system: engine.System, w_cm1: float, kx_cm1: np.ndarray) -> np.ndarray:
    """Vectorised evaluation of a whole rpp row (all kx at once).

    Returns the complex rpp row. Raises :class:`FastPathUnavailable` if the row is
    not eligible for the vectorised path (the caller should fall back to the
    per-point path, which evaluates each sample independently).
    """
    f_hz = float(np.asarray(w_cm1, dtype=float) * CM1_TO_HZ)
    system.initialize_sys(f_hz)
    z = _zeta_from_kx(w_cm1, kx_cm1)
    return _rpp_from_zeta(system, f_hz, z)


def compute_row_pointwise(system: engine.System, w_cm1: float, kx_cm1: np.ndarray) -> np.ndarray:
    """Per-point evaluation of a rpp row.

    Uses the same engine as :func:`compute_row_batched` but evaluates each kx
    sample on its own, so a single ill-conditioned point (one the batched path
    rejects for the whole row) is mapped to zero without affecting its neighbours.
    Numerically identical to the batched path everywhere the batched path succeeds.
    """
    f_hz = float(np.asarray(w_cm1, dtype=float) * CM1_TO_HZ)
    system.initialize_sys(f_hz)
    z = _zeta_from_kx(w_cm1, kx_cm1)
    out = np.empty(z.shape[0], dtype=np.complex128)
    for j in range(z.shape[0]):
        try:
            out[j] = _rpp_from_zeta(system, f_hz, z[j : j + 1])[0]
        except (FastPathUnavailable, np.linalg.LinAlgError, FloatingPointError, ValueError):
            out[j] = 0.0 + 0.0j
    return out


def _rpp_raw_from_zeta(system: engine.System, f_hz: float, z: np.ndarray) -> np.ndarray:
    """Complex rpp over a zeta array WITHOUT masking non-finite samples.

    :func:`_rpp_from_zeta` maps non-finite samples to zero, which would conflate a
    genuine reflection zero (numerator -> 0) with a pole (denominator -> 0). The
    mode finder needs the unmasked ratio so Newton iterations are not misled by the
    masking.
    """
    with np.errstate(all="ignore"):
        numer, denom = _rpp_numer_denom(system, f_hz, z)
        return numer / denom


def _inv_rpp_raw_from_zeta(system: engine.System, f_hz: float, z: np.ndarray) -> np.ndarray:
    """Reciprocal reflection coefficient ``1/rpp = denom/numer`` over a zeta array.

    The optical modes of the stack -- the bright bands in the Im(rpp) dispersion
    map -- are the **poles** of rpp, i.e. the zeros of the p-block determinant
    ``denom = G00*G22 - G02*G20`` (the dispersion relation for a self-sustaining
    field with no incident p-wave). Searching for ``1/rpp -> 0`` lands the Newton
    iteration on a true pole and is repelled by the reflection zeros (where
    ``|1/rpp|`` is large) -- the mirror image of :func:`_rpp_raw_from_zeta`. Using
    the reciprocal rather than ``denom`` alone keeps the residual dimensionless and
    on the same scale as :func:`find_rpp_zero`, and is naturally repelled by the
    spurious common roots where numerator and denominator vanish together.
    """
    with np.errstate(all="ignore"):
        numer, denom = _rpp_numer_denom(system, f_hz, z)
        return denom / numer


def _newton_complex_zero(
    values_at: Callable[[np.ndarray], np.ndarray],
    zeta0: complex,
    *,
    max_iter: int,
    tol: float,
    step_eps: float,
) -> tuple[complex, float, bool]:
    """Damped Newton search for a complex zero of an analytic ``values_at(zetas)``.

    Starting from ``zeta0`` a Newton iteration (central-difference derivative)
    refines the root into the complex plane to sub-grid precision. Any trial step
    that does not reduce ``|value|`` (or lands on a non-finite sample) is damped
    (halved, down to 1e-4), which keeps the search from running off toward a
    singularity of ``values_at``.

    ``values_at`` evaluates a *batch* of zeta values. The two derivative samples of
    an iteration are evaluated in one call, and the damped trial steps
    ``1, 1/2, 1/4, ...`` in one more (the first few eagerly, the rest only if all of
    those fail): the accepted step is exactly the one the sequential halving loop
    would pick, but the Python/NumPy call overhead (which dominates a 1-point
    transfer-matrix evaluation) is paid ~3x less often.

    Returns ``(zeta, residual, converged)``; ``converged`` is ``True`` only when
    the residual reached ``tol``.
    """
    z = complex(zeta0)
    fz = complex(values_at(np.array([z], dtype=np.complex128))[0])
    if not np.isfinite(fz):
        return z, float("inf"), False

    dampings = 0.5 ** np.arange(15)  # 1, 1/2, ..., 6.1e-5 (first value <= 1e-4 ends the loop)

    def first_acceptable(zs: np.ndarray, fs: np.ndarray, af: float) -> int | None:
        ok = np.isfinite(fs) & (np.abs(fs) <= af)
        idx = np.flatnonzero(ok)
        return int(idx[0]) if idx.size else None

    for _ in range(int(max_iter)):
        if abs(fz) < tol:
            return z, float(abs(fz)), True
        h = step_eps * max(abs(z), 1.0)
        fp, fm = values_at(np.array([z + h, z - h], dtype=np.complex128))
        df = (fp - fm) / (2.0 * h)
        if df == 0 or not np.isfinite(df):
            break
        step = fz / df
        af = abs(fz)

        pick = None
        for lo, hi in ((0, 4), (4, len(dampings))):
            zs = z - dampings[lo:hi] * step
            fs = values_at(zs.astype(np.complex128))
            k = first_acceptable(zs, fs, af)
            if k is not None:
                pick = (zs[k], fs[k])
                break
            last = (zs[-1], fs[-1])
        z_new, fz_new = pick if pick is not None else last
        if not np.isfinite(fz_new) or abs(fz_new) >= abs(fz):
            break
        z, fz = complex(z_new), complex(fz_new)

    return z, float(abs(fz)), bool(abs(fz) < tol)


def _batched_objective(
    raw_fn: Callable[[engine.System, float, np.ndarray], np.ndarray],
    system: engine.System,
    f_hz: float,
) -> Callable[[np.ndarray], np.ndarray]:
    def values_at(zetas: np.ndarray) -> np.ndarray:
        try:
            return np.asarray(raw_fn(system, f_hz, zetas), dtype=np.complex128)
        except (FastPathUnavailable, np.linalg.LinAlgError, FloatingPointError, ValueError):
            return np.full(zetas.shape, complex(np.nan, np.nan))

    return values_at


def find_rpp_zero(
    system: engine.System,
    f_hz: float,
    zeta0: complex,
    *,
    max_iter: int = 60,
    tol: float = 1e-9,
    step_eps: float = 1e-6,
) -> tuple[complex, float, bool]:
    """Locate the complex reduced wavevector ``zeta`` where ``rpp(zeta) = 0``.

    ``system`` must already be initialised at ``f_hz`` (e.g. via a prior row
    evaluation). Starting from ``zeta0`` -- typically the ``|rpp|`` minimum on the
    real kx grid -- a damped Newton iteration refines the reflection zero into the
    complex plane. Returns ``(zeta, residual, converged)`` with ``residual`` the
    final ``|rpp(zeta)|``.

    Note: this finds a *reflection zero* (Brewster/Zenneck-like), not a guided
    mode. The optical modes (the bright bands of the Im(rpp) map) are the poles of
    rpp -- use :func:`find_rpp_pole` for those.
    """
    return _newton_complex_zero(
        _batched_objective(_rpp_raw_from_zeta, system, f_hz),
        zeta0,
        max_iter=max_iter,
        tol=tol,
        step_eps=step_eps,
    )


def find_rpp_pole(
    system: engine.System,
    f_hz: float,
    zeta0: complex,
    *,
    max_iter: int = 60,
    tol: float = 1e-9,
    step_eps: float = 1e-6,
) -> tuple[complex, float, bool]:
    """Locate the complex reduced wavevector ``zeta`` of an optical mode.

    A mode is a **pole** of rpp -- equivalently a zero of ``1/rpp = denom/numer``,
    where ``denom`` is the p-polarised dispersion determinant. ``system`` must
    already be initialised at ``f_hz``. Starting from ``zeta0`` -- typically the
    ``|rpp|`` maximum on the real kx grid (the on-axis trace of the pole) -- a
    damped Newton iteration refines the pole into the complex plane to sub-grid
    precision; the real part is the mode wavevector and the imaginary part its loss.

    Returns ``(zeta, residual, converged)`` where ``residual`` is ``|1/rpp(zeta)|``
    and ``converged`` is ``True`` only when it reached ``tol``. A failed or
    diverging search returns ``converged=False`` so the caller can drop the point.
    """
    return _newton_complex_zero(
        _batched_objective(_inv_rpp_raw_from_zeta, system, f_hz),
        zeta0,
        max_iter=max_iter,
        tol=tol,
        step_eps=step_eps,
    )


# ---------------------------------------------------------------------------
# Many frequencies at once: grid seeds and a vectorised pole finder
# ---------------------------------------------------------------------------
#
# The scalar finder above evaluates one (omega, zeta) point per call, so its cost is
# dominated by Python/NumPy call overhead, not arithmetic. Because the layer code only
# reads ``epsilon[i, j]`` entries and ``alpha``, it runs unchanged on a SystemBatch
# whose tensors differ per sample, i.e. on arbitrary (omega_i, zeta_i) pairs in one
# call. The finders below exploit that: every frequency's Newton iteration advances
# in lock-step, so one iteration costs a few large batched evaluations instead of
# ``n_freq`` tiny ones.


def _batch_ratio(batch: engine.SystemBatch, f_hz: np.ndarray, z: np.ndarray, invert: bool) -> np.ndarray:
    """``rpp`` (or ``1/rpp``) for per-sample frequencies; failing samples give NaN."""
    try:
        with np.errstate(all="ignore"):
            numer, denom = _rpp_numer_denom(batch, f_hz, z)
            return denom / numer if invert else numer / denom
    except (FastPathUnavailable, np.linalg.LinAlgError, FloatingPointError, ValueError):
        out = np.full(z.shape, complex(np.nan, np.nan))
        if z.shape[0] == 1:
            return out
        half = z.shape[0] // 2  # isolate the failing sample(s) by bisection
        idx = np.arange(z.shape[0])
        for part in (idx[:half], idx[half:]):
            out[part] = _batch_ratio(batch.take(part), f_hz[part], z[part], invert)
        return out


def concat_batches(batches: list[engine.SystemBatch]) -> engine.SystemBatch:
    """Join single-sample SystemBatches (one per orientation) into one batch of N samples."""

    def cat(parts: list[engine.LayerBatch]) -> engine.LayerBatch:
        return engine.LayerBatch(
            parts[0].thick, parts[0].mu, np.concatenate([q.epsilon for q in parts], axis=2),
            all(q.is_isotropic for q in parts), all(q.is_block_diagonal for q in parts),
        )

    return engine.SystemBatch(
        cat([b.superstrate for b in batches]),
        cat([b.substrate for b in batches]),
        [cat([b.layers[i] for b in batches]) for i in range(len(batches[0].layers))],
    )


def rpp_grid(
    system: engine.System,
    w_values_cm1: np.ndarray,
    kx_values_cm1: np.ndarray,
    *,
    max_samples: int = 24_000,
) -> np.ndarray:
    """Complex rpp on a whole (omega, kx) grid, evaluated in a few large batches.

    Identical to calling :func:`compute_row_batched` row by row (non-finite samples
    map to zero) but amortises the per-call overhead over many rows.
    """
    w = np.asarray(w_values_cm1, dtype=float)
    kx = np.asarray(kx_values_cm1, dtype=float)
    nk = kx.size
    out = np.empty((w.size, nk), dtype=np.complex128)
    rows = max(1, max_samples // max(nk, 1))
    for start in range(0, w.size, rows):
        wc = w[start : start + rows]
        batch = system.batch_at(wc * CM1_TO_HZ)
        sample_row = np.repeat(np.arange(wc.size), nk)
        z = (np.tile(kx, wc.size) / wc[sample_row]).astype(np.complex128) + 1j * _ZETA_IMAG_REG
        r = _batch_ratio(batch.take(sample_row), wc[sample_row] * CM1_TO_HZ, z, invert=False)
        out[start : start + rows] = np.where(np.isfinite(r), r, 0.0).reshape(wc.size, nk)
    return out


def find_rpp_poles_batch(
    batch: engine.SystemBatch,
    f_hz: np.ndarray,
    zeta0: np.ndarray,
    *,
    max_iter: int = 30,
    tol: float = 1e-9,
    step_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised :func:`find_rpp_pole`: one pole per sample (frequency) in lock-step.

    Each sample follows exactly the damped Newton iteration of
    :func:`_newton_complex_zero` (same derivative stencil, same halving rule, same
    stopping tests); only the evaluations are batched across samples. Two further
    savings do not change the iteration:

    * the full step ``1`` is evaluated together with the derivative stencil
      ``zeta +- h`` of the point it leads to, so an accepted full step (the usual
      case) needs one batched call per iteration instead of two;
    * ``max_iter`` defaults to 30 rather than 60: every sample that converges at all
      does so within ~25 iterations (quadratic convergence; checked on ~3600
      frequencies of 150 random stacks), the rest only burn time creeping along
      damped steps.

    Returns ``(zeta, residual, converged)`` arrays.
    """
    f_hz = np.asarray(f_hz, dtype=float)
    z = np.asarray(zeta0, dtype=np.complex128).copy()
    n = z.size

    def values(index: np.ndarray, zs: np.ndarray) -> np.ndarray:
        return _batch_ratio(batch.take(index), f_hz[index], zs.astype(np.complex128), invert=True)

    def stencil_h(zs: np.ndarray) -> np.ndarray:
        return step_eps * np.maximum(np.abs(zs), 1.0)

    everyone = np.arange(n)
    fz = values(everyone, z)
    alive = np.isfinite(fz)  # samples still iterating
    started = alive.copy()
    converged = np.zeros(n, dtype=bool)
    fz = np.where(alive, fz, 0.0)
    fp = np.zeros(n, dtype=np.complex128)  # cached derivative stencil f(z+h), f(z-h)
    fm = np.zeros(n, dtype=np.complex128)
    have_stencil = np.zeros(n, dtype=bool)

    for _ in range(int(max_iter)):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        hit = np.abs(fz[idx]) < tol
        converged[idx[hit]] = True
        alive[idx[hit]] = False
        idx = idx[~hit]
        if idx.size == 0:
            break

        need = idx[~have_stencil[idx]]
        if need.size:
            h = stencil_h(z[need])
            both = values(np.concatenate([need, need]), np.concatenate([z[need] + h, z[need] - h]))
            fp[need], fm[need] = both[: need.size], both[need.size :]
            have_stencil[need] = True
        h = stencil_h(z[idx])
        df = (fp[idx] - fm[idx]) / (2.0 * h)
        ok = (df != 0) & np.isfinite(df)
        alive[idx[~ok]] = False
        idx, df = idx[ok], df[ok]
        if idx.size == 0:
            break
        step = fz[idx] / df
        have_stencil[idx] = False  # the stencil belongs to the point we are leaving

        # Damped trial steps 1, 1/2, 1/4, ... for the samples that have not yet found a
        # non-increasing step (the sequential halving rule, batched). Stage 0 also
        # evaluates the stencil at the trial point.
        z_new = np.empty(idx.size, dtype=np.complex128)
        f_new = np.empty(idx.size, dtype=np.complex128)
        fp_new = np.full(idx.size, np.nan, dtype=np.complex128)
        fm_new = np.full(idx.size, np.nan, dtype=np.complex128)
        pending = np.arange(idx.size)
        for k in range(15):
            zt = z[idx[pending]] - 0.5**k * step[pending]
            if k == 0:
                ht = stencil_h(zt)
                m = pending.size
                triple = values(np.tile(idx[pending], 3), np.concatenate([zt, zt + ht, zt - ht]))
                ft, fp_new[pending], fm_new[pending] = triple[:m], triple[m : 2 * m], triple[2 * m :]
            else:
                ft = values(idx[pending], zt)
            z_new[pending], f_new[pending] = zt, ft
            good = np.isfinite(ft) & (np.abs(ft) <= np.abs(fz[idx[pending]]))
            if k == 0:
                took_full_step = good.copy()
            pending = pending[~good]
            if pending.size == 0:
                break
        accept = np.isfinite(f_new) & (np.abs(f_new) < np.abs(fz[idx]))
        alive[idx[~accept]] = False
        z[idx[accept]], fz[idx[accept]] = z_new[accept], f_new[accept]
        # the stencil evaluated with the trial point is the next iteration's derivative
        # stencil when that (undamped, k = 0) trial was the accepted step
        use = accept & took_full_step & np.isfinite(fp_new) & np.isfinite(fm_new)
        fp[idx[use]], fm[idx[use]] = fp_new[use], fm_new[use]
        have_stencil[idx[use]] = True

    converged = (converged | (np.abs(fz) < tol)) & started  # a sample whose seed was non-finite never started
    residual = np.where(started, np.abs(fz), np.inf)
    return z, residual, converged
