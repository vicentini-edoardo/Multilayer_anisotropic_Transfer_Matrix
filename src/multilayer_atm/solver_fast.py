"""In-house transfer-matrix engine for Im(rpp) rows.

This module is the numerical core of the solver. It implements the generalized
4x4 transfer-matrix algorithm (Passler & Paarmann 2017) directly in NumPy,
batched over the in-plane momentum axis: all kx samples of a single
frequency/angle row are processed in one set of array operations. It depends only
on the lightweight :mod:`multilayer_atm.engine` containers (no pyGTM at runtime);
pyGTM is retained solely as an independent validation reference for the tests.

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
    r = np.sqrt(D)
    flip = (np.abs(r.imag) >= _ZERO_THR) & (r.imag < 0.0)
    qt = np.where(flip, -r, r)

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
    gs = _batched_gamma_star(system, f_hz, z)
    denom = gs[:, 0, 0] * gs[:, 2, 2] - gs[:, 0, 2] * gs[:, 2, 0]
    numer = gs[:, 1, 0] * gs[:, 2, 2] - gs[:, 1, 2] * gs[:, 2, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
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
