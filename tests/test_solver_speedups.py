"""Regression tests for the solver speed-ups.

Every optimisation must reproduce the original algorithm:

* the layer-propagation shortcuts (closed forms, 4x2 block propagation) against the
  original full ``A K A^-1`` product of :func:`solver_fast._batched_gamma_star`;
* the isofrequency symmetry reduction against a brute-force sweep of every azimuth;
* the vectorised permittivity pre-evaluation against per-row evaluation;
* the batched Newton iteration against the original one-point-at-a-time loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from multilayer_atm import solver, solver_fast
from multilayer_atm.materials import CM1_TO_HZ
from multilayer_atm.models import DopingSpec, LayerSpec, StackSpec


def _stack(*layers: tuple[str, float, tuple[float, float, float]]) -> StackSpec:
    return StackSpec.from_layers([LayerSpec(m, t, e, DopingSpec()) for m, t, e in layers])


# (label, stack). Covers: isotropic-only, block-diagonal (beta = 0), tilted (eig), anisotropic
# boundaries, and a mix of all of them.
STACKS = {
    "isotropic_film": _stack(("vacuum", 0, (0, 0, 0)), ("SiC3C", 100e-9, (0, 0, 0)), ("Si", 0, (0, 0, 0))),
    "block_in_plane_rotated": _stack(
        ("vacuum", 0, (0, 0, 0)), ("MoO3", 150e-9, (20, 0, 0)), ("SiC3C", 1e-6, (0, 0, 0)), ("Si", 0, (0, 0, 0))
    ),
    "uniaxial_c_axis_out_of_plane": _stack(("vacuum", 0, (0, 0, 0)), ("hBN", 80e-9, (0, 0, 0)), ("Si", 0, (0, 0, 0))),
    "tilted_biaxial": _stack(
        ("vacuum", 0, (0, 0, 0)), ("hBN", 100e-9, (10, 20, 30)), ("MoO3", 150e-9, (5, 15, 25)), ("SiC3C", 1e-6, (0, 0, 0))
    ),
    "anisotropic_boundaries": _stack(
        ("hBN", 0, (0, 0, 0)), ("SiC3C", 60e-9, (0, 0, 0)), ("MoO3", 120e-9, (33, 0, 0)), ("MoO3", 0, (40, 25, 10))
    ),
}


def _reference_rpp(system, f_hz, z):
    """The original implementation: full A K A^-1 product, 4x4 inverse, gamma_star minors."""
    gs = solver_fast._batched_gamma_star(system, f_hz, z)
    denom = gs[:, 0, 0] * gs[:, 2, 2] - gs[:, 0, 2] * gs[:, 2, 0]
    numer = gs[:, 1, 0] * gs[:, 2, 2] - gs[:, 1, 2] * gs[:, 2, 0]
    with np.errstate(all="ignore"):
        out = numer / denom
    return np.where(np.isfinite(out), out, 0.0)


@pytest.mark.parametrize("label", list(STACKS))
@pytest.mark.parametrize("w_cm1", [550.0, 850.0, 1150.0])
def test_fast_propagation_matches_original_algorithm(label: str, w_cm1: float) -> None:
    system = solver._build_system(STACKS[label])
    f_hz = w_cm1 * CM1_TO_HZ
    system.initialize_sys(f_hz)
    # include kx = 0 (degenerate uniaxial modes) and strongly evanescent samples
    kx = np.concatenate([[0.0], np.linspace(0.0, 4e4, 150)])
    z = solver_fast._zeta_from_kx(w_cm1, kx)

    new = solver_fast._rpp_from_zeta(system, f_hz, z)
    old = _reference_rpp(system, f_hz, z)

    scale = np.max(np.abs(old))
    assert np.max(np.abs(new - old)) < 1e-8 * scale


@pytest.mark.parametrize("label", list(STACKS))
def test_closed_forms_match_generic_eig_path(label: str) -> None:
    """force_eig=True bypasses the isotropic / block-diagonal closed forms."""
    system = solver._build_system(STACKS[label])
    w = 900.0
    f_hz = w * CM1_TO_HZ
    system.initialize_sys(f_hz)
    z = solver_fast._zeta_from_kx(w, np.linspace(0.0, 3e4, 120))
    with np.errstate(all="ignore"):
        n1, d1 = solver_fast._rpp_numer_denom(system, f_hz, z)
        n2, d2 = solver_fast._rpp_numer_denom(system, f_hz, z, force_eig=True)
    assert np.max(np.abs(n1 / d1 - n2 / d2)) < 1e-8 * np.max(np.abs(n2 / d2))


def test_block_diagonal_classification() -> None:
    system = solver._build_system(STACKS["tilted_biaxial"])
    system.initialize_sys(850.0 * CM1_TO_HZ)
    hbn, moo3 = system.layers[0], system.layers[1]
    assert not hbn.is_isotropic and not hbn.is_block_diagonal  # beta = 20 deg
    assert not moo3.is_block_diagonal
    assert system.substrate.is_isotropic

    system = solver._build_system(STACKS["block_in_plane_rotated"])
    system.initialize_sys(850.0 * CM1_TO_HZ)
    assert system.layers[0].is_block_diagonal and not system.layers[0].is_isotropic


def test_prepare_frequencies_matches_per_row_evaluation() -> None:
    stack = STACKS["tilted_biaxial"]
    w_values = np.linspace(700.0, 1000.0, 7)
    plain = solver._build_system(stack)
    prepared = solver._build_system(stack)
    prepared.prepare_frequencies(w_values * CM1_TO_HZ)
    kx = np.linspace(0.0, 2e4, 40)
    for w in w_values:
        a = solver._compute_row(plain, float(w), kx, True)
        b = solver._compute_row(prepared, float(w), kx, True)
        assert np.max(np.abs(a - b)) < 1e-12


# --- isofrequency azimuth symmetry -----------------------------------------------------------


def _brute_force_isofreq(stack: StackSpec, w0: float, kx: np.ndarray, phis_deg: np.ndarray) -> np.ndarray:
    rows = []
    for phi in phis_deg:
        system = solver._build_system(stack.with_interior_gamma_offset(float(phi)))
        rows.append(solver._compute_row(system, w0, kx, True))
    return np.array(rows)


@pytest.mark.parametrize(
    "stack, expect_mirror",
    [
        # beta = 0 and alpha + gamma multiple of 90 deg -> pi periodic AND mirror symmetric
        (_stack(("vacuum", 0, (0, 0, 0)), ("MoO3", 150e-9, (0, 0, 0)), ("SiC3C", 1e-6, (0, 0, 0))), True),
        (_stack(("vacuum", 0, (0, 0, 0)), ("MoO3", 150e-9, (90, 0, 0)), ("SiC3C", 1e-6, (0, 0, 90))), True),
        # in-plane rotated crystal -> only pi periodic
        (_stack(("vacuum", 0, (0, 0, 0)), ("MoO3", 150e-9, (20, 0, 0)), ("SiC3C", 1e-6, (0, 0, 0))), False),
        # tilted crystal -> only pi periodic
        (_stack(("vacuum", 0, (0, 0, 0)), ("hBN", 100e-9, (0, 30, 0)), ("Si", 0, (0, 0, 0))), False),
    ],
)
def test_isofrequency_symmetry_reduction_is_exact(stack: StackSpec, expect_mirror: bool) -> None:
    assert solver._mirror_symmetric_about_phi0(stack.enforce_boundary_layers()) is expect_mirror

    nphi, w0 = 24, 850.0
    kx_max, nk = 3e4, 30
    phi, kx, rpp = solver.compute_isofreq_map(
        stack, w0, 0.0, kx_max, nk, nphi, workers=1, fast=True, return_complex=True
    )
    brute = _brute_force_isofreq(stack, w0, kx, np.rad2deg(phi))
    assert np.max(np.abs(rpp - brute)) < 1e-9 * np.max(np.abs(brute))


def test_isofrequency_symmetry_reduces_work_and_handles_partial_sweeps() -> None:
    stack = STACKS["tilted_biaxial"]
    offs, inverse = solver._unique_azimuths(np.arange(0.0, 360.0, 15.0), mirror=False)
    assert offs.size == 12 and inverse.size == 24  # pi periodicity: half the rows
    offs, inverse = solver._unique_azimuths(np.arange(0.0, 360.0, 15.0), mirror=True)
    assert offs.size == 7  # 0, 15, ..., 90 deg
    offs, inverse = solver._unique_azimuths(np.zeros(16), mirror=False)  # non-global sweep
    assert offs.size == 1

    # a partial, unevenly-covered sweep must still be exact
    phi, kx, rpp = solver.compute_isofreq_map(
        stack, 850.0, 0.0, 2e4, 20, 10, phi_min_deg=-40.0, phi_max_deg=200.0, workers=1, fast=True, return_complex=True
    )
    brute = _brute_force_isofreq(stack, 850.0, kx, np.rad2deg(phi))
    assert np.max(np.abs(rpp - brute)) < 1e-9 * np.max(np.abs(brute))


def test_isofrequency_without_global_sweep_is_constant() -> None:
    stack = STACKS["block_in_plane_rotated"]
    _phi, _kx, im = solver.compute_isofreq_map(stack, 850.0, 0.0, 2e4, 16, 12, global_phi_sweep=False, workers=1, fast=True)
    assert np.all(im == im[0:1, :])


# --- mode finder ---------------------------------------------------------------------------


def _sequential_newton(value_at, zeta0, *, max_iter=60, tol=1e-9, step_eps=1e-6):
    """The original one-point-at-a-time damped Newton loop (reference)."""
    z = complex(zeta0)
    fz = value_at(z)
    if not np.isfinite(fz):
        return z, float("inf"), False
    for _ in range(int(max_iter)):
        if abs(fz) < tol:
            return z, float(abs(fz)), True
        h = step_eps * max(abs(z), 1.0)
        df = (value_at(z + h) - value_at(z - h)) / (2.0 * h)
        if df == 0 or not np.isfinite(df):
            break
        step = fz / df
        damp = 1.0
        z_new = z - step
        fz_new = value_at(z_new)
        while (not np.isfinite(fz_new) or abs(fz_new) > abs(fz)) and damp > 1e-4:
            damp *= 0.5
            z_new = z - damp * step
            fz_new = value_at(z_new)
        if not np.isfinite(fz_new) or abs(fz_new) >= abs(fz):
            break
        z, fz = z_new, fz_new
    return z, float(abs(fz)), bool(abs(fz) < tol)


def test_batched_newton_follows_the_sequential_iteration() -> None:
    # damped steps are exercised by a function with a nearby singularity
    funcs = [
        lambda z: (z - (2.0 + 0.5j)) * (1.0 + 0.2 * z),
        lambda z: np.tan(z) - 1.0,
        lambda z: z**3 - 2.0 * z + 2.0,  # classic Newton cycle at z0 = 0
    ]
    for fn in funcs:
        for z0 in (0.1 + 0j, 1.3 - 0.4j, 0.0 + 0j, -2.0 + 1.0j):
            seq = _sequential_newton(lambda z: complex(fn(z)), z0)
            bat = solver_fast._newton_complex_zero(
                lambda zs: np.asarray(fn(zs), dtype=np.complex128), z0, max_iter=60, tol=1e-9, step_eps=1e-6
            )
            assert bat[2] == seq[2]
            assert abs(bat[0] - seq[0]) < 1e-12
            assert abs(bat[1] - seq[1]) < 1e-12


def test_mode_trace_matches_original_newton_on_a_real_stack() -> None:
    stack = STACKS["isotropic_film"]
    w, nk = 880.0, 120
    kx = np.linspace(0.0, 3e4, nk)
    system = solver._build_system(stack)
    row = solver._compute_row(system, w, kx, True)
    j0 = int(np.argmax(np.abs(row)))
    zeta0 = complex(kx[j0] / w, 0.0)
    f_hz = w * CM1_TO_HZ

    def value_at(zeta: complex) -> complex:
        return complex(solver_fast._inv_rpp_raw_from_zeta(system, f_hz, np.array([zeta], dtype=np.complex128))[0])

    seq = _sequential_newton(value_at, zeta0)
    new = solver_fast.find_rpp_pole(system, f_hz, zeta0)
    assert new[2] == seq[2]
    assert abs(new[0] - seq[0]) < 1e-9 * max(abs(seq[0]), 1.0)


# --- worker heuristic ---------------------------------------------------------------------------


def test_small_jobs_run_serially_large_jobs_keep_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("multiprocessing.get_start_method", lambda allow_none=False: "fork")
    assert solver._resolve_workers(1, 100.0) == 1
    assert solver._resolve_workers(4, 0.01) == 1
    assert solver._resolve_workers(4, 5.0) == 4
    monkeypatch.setattr("multiprocessing.get_start_method", lambda allow_none=False: "spawn")
    assert solver._resolve_workers(4, 0.3) == 1  # spawn start-up is far costlier
    assert solver._resolve_workers(4, 5.0) == 4


def test_parallel_and_serial_results_agree() -> None:
    stack = STACKS["tilted_biaxial"]
    kw = dict(w_min=700.0, w_max=1000.0, nw=24, kx_min=0.0, kx_max=3e4, nk=40, fast=True)
    _w, _k, serial = solver.compute_rpp_map(stack, workers=1, **kw)
    # force the pool even though the job is tiny
    original = solver._resolve_workers
    solver._resolve_workers = lambda requested, est: requested
    try:
        _w, _k, parallel = solver.compute_rpp_map(stack, workers=2, **kw)
    finally:
        solver._resolve_workers = original
    assert np.max(np.abs(serial - parallel)) < 1e-12


# --- multi-frequency batch and vectorised mode finder ------------------------------------------


def _modes_legacy(stack: StackSpec, w_values, kx_values):
    """Per-frequency scalar mode search (the previous implementation)."""
    system = solver._build_system(stack.enforce_boundary_layers())
    out = [solver._find_mode_for_frequency(system, float(w), kx_values, True) for w in w_values]
    return (
        np.array([o[0] for o in out]),
        np.array([o[1] for o in out]),
        np.array([o[2] for o in out]),
    )


@pytest.mark.parametrize("label", ["isotropic_film", "block_in_plane_rotated", "tilted_biaxial"])
def test_rpp_grid_matches_row_by_row(label: str) -> None:
    system = solver._build_system(STACKS[label].enforce_boundary_layers())
    w = np.linspace(700.0, 1000.0, 9)
    kx = np.linspace(0.0, 2e4, 33)
    grid = solver_fast.rpp_grid(system, w, kx, max_samples=100)  # several chunks
    rows = np.array([solver._compute_row(system, float(wi), kx, True) for wi in w])
    assert np.max(np.abs(grid - rows)) < 1e-9 * np.max(np.abs(rows))


@pytest.mark.parametrize("label", ["isotropic_film", "block_in_plane_rotated", "tilted_biaxial"])
def test_vectorised_mode_finder_matches_scalar_finder(label: str, monkeypatch) -> None:
    # the trace repair (jump removal) intentionally differs from the raw per-frequency finder
    monkeypatch.setattr(solver, "_repair_mode_trace", lambda system, f, w, k, z, r, c, **kw: (z, r, c))
    stack = STACKS[label]
    w0, w1, nw, kmax, nk = 780.0, 1000.0, 30, 4e4, 120
    w, kx_mode, residual, converged = solver.compute_mode_dispersion(stack, w0, w1, nw, 0.0, kmax, nk, workers=1, fast=True)
    kx_old, res_old, conv_old = _modes_legacy(stack, w, np.linspace(0.0, kmax, nk))
    # "Converged" poles with |Im kx| >> Re kx or far outside the grid come from the
    # overflow/noise regime of very evanescent samples; there either implementation
    # is arbitrary, so the comparison is on physical poles only.
    def physical(kx, ok):
        return ok & (np.abs(kx.real) < kmax) & (np.abs(kx.imag) < 0.5 * np.abs(kx.real))

    new_phys, old_phys = physical(kx_mode, converged), physical(kx_old, conv_old)
    assert new_phys.any()
    assert np.array_equal(new_phys, old_phys)
    assert np.max(np.abs(kx_mode[new_phys] - kx_old[new_phys]) / np.abs(kx_old[new_phys])) < 1e-9


def test_mode_seed_is_reused_from_the_dispersion_map_exactly() -> None:
    stack = STACKS["block_in_plane_rotated"]
    args = (780.0, 1000.0, 24, 0.0, 4e4, 90)
    solver._SEED_CACHE.clear()
    cold = solver.compute_mode_dispersion(stack, *args, workers=1, fast=True)
    assert len(solver._SEED_CACHE) == 0  # a mode trace alone does not populate the cache

    solver.compute_rpp_map(stack, *args, workers=1, fast=True)
    assert len(solver._SEED_CACHE) == 1
    called = []
    original = solver_fast.rpp_grid
    solver_fast.rpp_grid = lambda *a, **k: called.append(1) or original(*a, **k)
    try:
        warm = solver.compute_mode_dispersion(stack, *args, workers=1, fast=True)
    finally:
        solver_fast.rpp_grid = original
    assert not called  # seed grid not re-evaluated
    assert np.array_equal(cold[3], warm[3])
    assert np.max(np.abs(cold[1][cold[3]] - warm[1][warm[3]])) < 1e-9

    # a different stack or grid must not hit the cache
    other = solver.compute_mode_dispersion(STACKS["isotropic_film"], *args, workers=1, fast=True)
    assert other[1].shape == cold[1].shape
    solver._SEED_CACHE.clear()


def test_parallel_mode_blocks_match_serial() -> None:
    stack = STACKS["block_in_plane_rotated"]
    args = (780.0, 1000.0, 24, 0.0, 4e4, 90)
    solver._SEED_CACHE.clear()
    serial = solver.compute_mode_dispersion(stack, *args, workers=1, fast=True)
    original = solver._resolve_workers
    solver._resolve_workers = lambda requested, est: requested  # force the pool for this small job
    try:
        parallel = solver.compute_mode_dispersion(stack, *args, workers=3, fast=True)
    finally:
        solver._resolve_workers = original
    assert np.array_equal(serial[3], parallel[3])
    assert np.max(np.abs(serial[1][serial[3]] - parallel[1][parallel[3]])) < 1e-8


def test_seed_abs_rpp_argument_is_validated_and_used() -> None:
    stack = STACKS["isotropic_film"]
    args = (780.0, 1000.0, 12, 0.0, 3e4, 60)
    system = solver._build_system(stack.enforce_boundary_layers())
    w = np.linspace(780.0, 1000.0, 12)
    kx = np.linspace(0.0, 3e4, 60)
    seed = np.abs(solver_fast.rpp_grid(system, w, kx))
    solver._SEED_CACHE.clear()
    a = solver.compute_mode_dispersion(stack, *args, workers=1, fast=True)
    b = solver.compute_mode_dispersion(stack, *args, workers=1, fast=True, seed_abs_rpp=seed)
    assert np.array_equal(a[3], b[3])
    with pytest.raises(ValueError):
        solver.compute_mode_dispersion(stack, *args, workers=1, fast=True, seed_abs_rpp=seed[:, :10])


def test_batched_newton_handles_failing_samples_independently() -> None:
    """One non-finite sample must not take the whole batch (or its neighbours) down."""
    system = solver._build_system(STACKS["tilted_biaxial"].enforce_boundary_layers())
    w = np.array([800.0, 850.0, 900.0])
    f = w * CM1_TO_HZ
    batch = system.batch_at(f)
    z = np.array([5.0 + 0.0j, np.nan + 0.0j, 6.0 + 0.0j])
    zeta, residual, converged = solver_fast.find_rpp_poles_batch(batch, f, z)
    assert not converged[1] and residual[1] == np.inf
    system.initialize_sys(float(f[0]))
    ref0 = solver_fast.find_rpp_pole(system, float(f[0]), z[0])
    assert abs(zeta[0] - ref0[0]) < 1e-9 * max(abs(ref0[0]), 1.0)
    assert converged[0] == ref0[2]


def test_system_batch_flags_and_tensors_match_scalar_evaluation() -> None:
    system = solver._build_system(STACKS["anisotropic_boundaries"].enforce_boundary_layers())
    f = np.array([600.0, 900.0, 1200.0]) * CM1_TO_HZ
    batch = system.batch_at(f)
    for i, fi in enumerate(f):
        system.initialize_sys(float(fi))
        for single, many in zip((system.superstrate, system.substrate, *system.layers), (batch.superstrate, batch.substrate, *batch.layers)):
            assert np.max(np.abs(single.epsilon - many.epsilon[:, :, i])) < 1e-12 * np.max(np.abs(single.epsilon))
    assert not batch.layers[1].is_isotropic  # MoO3 stays anisotropic at every frequency


# ---------------------------------------------------------------------------------------
# Independent validation and thick-layer stability
# ---------------------------------------------------------------------------------------

def test_matches_independent_berreman_reference() -> None:
    """Different algorithm (scipy expm + eigen-decomposed boundaries), tilted anisotropic layers."""
    from berreman_reference import rpp_ref

    rng = np.random.default_rng(7)
    mats = ["hBN", "MoO3", "SiC3C", "alpha_quartz", "calcite", "ZnO", "Au"]
    for _ in range(12):
        layers = [("vacuum", 0.0, (0, 0, 0))]
        for _ in range(rng.integers(1, 4)):
            layers.append((str(rng.choice(mats)), float(10 ** rng.uniform(-8, -6.7)), tuple(rng.uniform(0, 180, 3))))
        layers.append((str(rng.choice(mats)), 0.0, tuple(rng.uniform(0, 180, 3))))
        system = solver._build_system(_stack(*layers))
        w = float(rng.uniform(400, 1700))
        system.initialize_sys(w * CM1_TO_HZ)
        kx = np.concatenate([[0.0], rng.uniform(0, 5 * w, 8)])
        fast = solver_fast.compute_row_batched(system, w, kx)
        z = kx / w + 1j * solver_fast._ZETA_IMAG_REG
        ref = np.array([-rpp_ref(system, w * CM1_TO_HZ, zi) for zi in z])
        assert np.max(np.abs(fast - ref) / (1 + np.abs(ref))) < 1e-9


@pytest.mark.parametrize("material", ["MoO3", "hBN", "alpha_quartz", "SiC3C", "Au"])
@pytest.mark.parametrize("euler", [(0, 0, 0), (35, 0, 0), (10, 20, 30)])
def test_thick_evanescent_layers_are_stable(material: str, euler: tuple[float, float, float]) -> None:
    """Splitting a layer in two (or adding a zero-thickness one) must not change rpp, even when
    k0*d*zeta reaches hundreds (growth e^X far beyond double precision of a naive product)."""
    vac, si = ("vacuum", 0.0, (0, 0, 0)), ("Si", 0.0, (0, 0, 0))
    d = 20e-6
    one = solver._build_system(_stack(vac, (material, d, euler), si))
    two = solver._build_system(_stack(vac, (material, d / 2, euler), (material, d / 2, euler), si))
    w = 900.0
    kx = np.linspace(500.0, 2e5, 60)  # k0*d*zeta up to ~700 -> exp overflow if mishandled
    a = solver_fast.compute_row_batched(one, w, kx)
    b = solver_fast.compute_row_batched(two, w, kx)
    assert np.all(np.isfinite(a)) and np.all(a != 0)
    assert np.max(np.abs(a - b)) < 1e-9


def test_mode_trace_has_no_jumps_and_stays_inside_the_kx_range():
    stack = _stack(
        ("vacuum", 0, (0, 0, 0)), ("hBN", 50e-9, (0, 0, 0)), ("SiC3C", 200e-9, (0, 0, 0)), ("Si", 0, (0, 0, 0))
    )
    kmax = 3e5
    w, kx, _res, ok = solver.compute_mode_dispersion(stack, 1300, 1650, 200, 0.0, kmax, 400, workers=1, fast=True)
    re = np.where(ok, kx.real, np.nan)
    assert ok.sum() > 20
    assert np.nanmax(re) <= kmax and np.nanmin(re) >= 0.0
    assert np.nanmax(np.abs(np.diff(re))) < 0.05 * kmax
