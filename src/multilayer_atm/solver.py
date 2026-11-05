"""Numerical solver routines for dispersion and isofrequency Im(rpp) maps."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import json
import os
import sys
from collections import OrderedDict
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import warnings

import numpy as np

from . import engine, solver_fast
from .materials import CM1_TO_HZ, axes_for_material
from .models import LayerSpec, StackSpec


ProgressCallback = Optional[Callable[[float, str], None]]

# Exceptions that legitimately signal a singular/ill-conditioned point in the
# transfer-matrix evaluation. These are mapped to a zero reflectivity sample.
# Anything outside this set (KeyError for an unknown material, AttributeError,
# TypeError, ...) is a genuine bug and is allowed to propagate so it surfaces to
# the user instead of producing a misleading all-zero map.
_NUMERICAL_ERRORS = (np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError, ValueError)

_WORKER_STACK_PAYLOAD: Optional[Dict[str, object]] = None
_WORKER_KX_VALUES_CM1: Optional[np.ndarray] = None
_WORKER_SYSTEM_CACHE: Dict[float, engine.System] = {}
_WORKER_FAST: bool = False
_WORKER_PREPARE_HZ: Optional[np.ndarray] = None


def cm1_to_hz(values_cm1: np.ndarray | float) -> np.ndarray:
    """Convert wavenumbers in cm⁻¹ to frequencies in Hz."""
    return np.asarray(values_cm1, dtype=float) * CM1_TO_HZ


def zeta_from_kx_w(kx_cm1: np.ndarray | float, w_cm1: np.ndarray | float) -> np.ndarray:
    """Convert in-plane momentum and frequency to the reduced wavevector zeta."""
    w = np.asarray(w_cm1, dtype=float)
    kx = np.asarray(kx_cm1, dtype=float)
    return kx / w


def passler_euler_radians(euler_deg: Sequence[float]) -> Tuple[float, float, float]:
    """Map Passler (alpha, beta, gamma) degrees to engine (theta, phi, psi) radians."""
    alpha, beta, gamma = euler_deg
    theta = np.deg2rad(beta)
    phi = np.deg2rad(alpha)
    psi = np.deg2rad(gamma)
    return float(theta), float(phi), float(psi)


def _prepare_grid_hz(w_values_cm1: np.ndarray) -> Optional[np.ndarray]:
    """Frequencies (Hz) worth pre-evaluating, or ``None`` for a single-frequency sweep.

    A multi-frequency grid (dispersion map, mode trace) evaluates each layer's
    permittivity once per row; doing it in one vectorised call up front removes
    that per-row overhead. An isofrequency sweep reuses one frequency (already
    memoised by the material catalog), so nothing is prepared there.
    """
    f = np.unique(np.asarray(w_values_cm1, dtype=float) * CM1_TO_HZ)
    return f if f.size > 1 else None


# Rough per-sample cost (microseconds) of the batched row solver, measured on a
# desktop CPU: base overhead + isotropic / block-diagonal (closed form) / general
# (batched eig) interior layers. Only the order of magnitude matters: it decides
# whether process-pool start-up can pay for itself.
_US_BASE, _US_ISO, _US_BLOCK, _US_EIG = 0.8, 0.3, 1.0, 10.5
_PARALLEL_MIN_SERIAL_S_FORK = 0.15
_PARALLEL_MIN_SERIAL_S_SPAWN = 0.6  # spawn re-imports numpy/materials in every worker


def _estimate_serial_seconds(system: engine.System, f_hz: float, n_rows: int, nk: int, extra_us_per_row: float = 0.0) -> float:
    system.initialize_sys(f_hz)
    per_point = _US_BASE
    for layer in system.layers:
        if layer.is_isotropic:
            per_point += _US_ISO
        elif layer.is_block_diagonal:
            per_point += _US_BLOCK
        else:
            per_point += _US_EIG
    return n_rows * (nk * per_point + extra_us_per_row) * 1e-6


def _resolve_workers(requested: int, estimated_serial_s: float) -> int:
    """Run serially when the job is shorter than what a worker pool costs to start.

    After the closed-form / symmetry speed-ups most presets finish in tens of
    milliseconds, whereas starting a ProcessPoolExecutor costs ~0.05 s with ``fork``
    and several hundred ms with ``spawn`` (macOS, Windows). Results are identical.
    """
    if requested <= 1:
        return 1
    import multiprocessing

    method = multiprocessing.get_start_method(allow_none=True) or ("fork" if sys.platform.startswith("linux") else "spawn")
    threshold = _PARALLEL_MIN_SERIAL_S_FORK if method == "fork" else _PARALLEL_MIN_SERIAL_S_SPAWN
    return requested if estimated_serial_s >= threshold else 1


def _default_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(4, cpu))


def _build_system(stack_spec: StackSpec, custom_materials: Mapping[str, Mapping[str, Any]] | None = None) -> engine.System:
    stack = stack_spec.enforce_boundary_layers()
    system = engine.System()

    layers = list(stack.layers)
    super_layer = _build_layer(layers[0], custom_materials=custom_materials)
    sub_layer = _build_layer(layers[-1], custom_materials=custom_materials)
    system.set_superstrate(super_layer)
    system.set_substrate(sub_layer)

    for mid in layers[1:-1]:
        system.add_layer(_build_layer(mid, custom_materials=custom_materials))

    return system


def _build_layer(layer_spec: LayerSpec, custom_materials: Mapping[str, Mapping[str, Any]] | None = None) -> engine.Layer:
    axes = axes_for_material(layer_spec.material, layer_spec.doping, custom_materials=custom_materials)
    theta, phi, psi = passler_euler_radians(layer_spec.euler_deg)
    return engine.Layer(
        thickness=float(layer_spec.thickness_m),
        epsilon1=axes.fx,
        epsilon2=axes.fy,
        epsilon3=axes.fz,
        theta=theta,
        phi=phi,
        psi=psi,
    )


def _compute_row(system: engine.System, w_cm1: float, kx_cm1: np.ndarray, fast: bool) -> np.ndarray:
    """Compute one Im(rpp) row with the in-house transfer-matrix engine.

    ``fast`` selects the vectorised whole-row evaluation; otherwise each kx sample
    is evaluated independently. Both use the same engine and agree to floating-point
    round-off. When the vectorised path rejects a row (an ill-conditioned sample
    that breaks the batched mode sort) it transparently falls back to the per-point
    path, so enabling ``fast`` never changes the result.
    """
    if not fast:
        return solver_fast.compute_row_pointwise(system, w_cm1, kx_cm1)
    try:
        return solver_fast.compute_row_batched(system, w_cm1, kx_cm1)
    except (solver_fast.FastPathUnavailable, *_NUMERICAL_ERRORS):
        return solver_fast.compute_row_pointwise(system, w_cm1, kx_cm1)


def _stack_to_worker_payload(stack_spec: StackSpec, custom_materials: Mapping[str, Mapping[str, Any]] | None = None) -> Dict[str, object]:
    return {
        "layers": [
            {
                "material": ls.material,
                "thickness_m": ls.thickness_m,
                "euler_deg": tuple(ls.euler_deg),
                "doping": {
                    "enabled": ls.doping.enabled,
                    "wp_cm1": ls.doping.wp_cm1,
                    "gp_cm1": ls.doping.gp_cm1,
                },
            }
            for ls in stack_spec.layers
        ],
        "custom_materials": dict(custom_materials or {}),
    }


def _worker_init(
    stack_payload: Dict[str, object],
    kx_values_cm1: np.ndarray,
    fast: bool,
    prepare_hz: Optional[np.ndarray] = None,
) -> None:
    global _WORKER_STACK_PAYLOAD, _WORKER_KX_VALUES_CM1, _WORKER_SYSTEM_CACHE, _WORKER_FAST, _WORKER_PREPARE_HZ
    _WORKER_PREPARE_HZ = None if prepare_hz is None else np.asarray(prepare_hz, dtype=float)
    _WORKER_STACK_PAYLOAD = stack_payload
    _WORKER_KX_VALUES_CM1 = np.asarray(kx_values_cm1, dtype=float)
    _WORKER_SYSTEM_CACHE = {}
    _WORKER_FAST = bool(fast)


def _get_worker_system(phi_offset_deg: float) -> engine.System:
    if _WORKER_STACK_PAYLOAD is None:
        raise RuntimeError("Worker stack payload not initialized.")

    key = float(phi_offset_deg)
    cached = _WORKER_SYSTEM_CACHE.get(key)
    if cached is not None:
        return cached

    stack = _deserialize_stack(_WORKER_STACK_PAYLOAD)
    if key != 0.0:
        stack = stack.with_interior_gamma_offset(key)
    system = _build_system(stack, custom_materials=_WORKER_STACK_PAYLOAD.get("custom_materials", {}))
    if _WORKER_PREPARE_HZ is not None:
        system.prepare_frequencies(_WORKER_PREPARE_HZ)
    _WORKER_SYSTEM_CACHE[key] = system
    return system


def _worker_compute_row_task(task: Tuple[int, float, float]) -> Tuple[int, np.ndarray]:
    row_index, w_cm1, phi_offset_deg = task
    if _WORKER_KX_VALUES_CM1 is None:
        raise RuntimeError("Worker kx values not initialized.")
    system = _get_worker_system(phi_offset_deg)
    row = _compute_row(system, w_cm1, _WORKER_KX_VALUES_CM1, _WORKER_FAST)
    return row_index, row


def _find_mode_for_frequency(
    system: engine.System, w_cm1: float, kx_values_cm1: np.ndarray, fast: bool
) -> Tuple[complex, float, bool]:
    """Locate the complex kx of the optical mode (pole of rpp) at one frequency.

    The optical modes are the poles of rpp (the bright bands of the Im(rpp) map):
    the dispersion determinant vanishes there. The real-kx row gives the starting
    guess (the kx of *maximum* |rpp|, the on-axis trace of the pole), which is then
    refined into the complex plane by :func:`solver_fast.find_rpp_pole`.
    Returns ``(kx_complex_cm1, residual, converged)``; ``kx_complex_cm1`` is
    ``zeta * w`` in cm⁻¹ (real part = mode wavevector, imaginary part = loss).
    """
    f_hz = float(w_cm1 * CM1_TO_HZ)
    # Evaluates the row (which initialises the system at f_hz) and gives the
    # coarse grid guess for the pole (the |rpp| peak on the real axis).
    row = _compute_row(system, w_cm1, kx_values_cm1, fast)
    mag = np.abs(np.asarray(row))
    finite = np.isfinite(mag)
    if not np.any(finite):
        return complex(np.nan, np.nan), float("inf"), False
    j0 = int(np.argmax(np.where(finite, mag, -np.inf)))
    zeta0 = complex(float(kx_values_cm1[j0]) / float(w_cm1), 0.0)
    zeta, residual, converged = solver_fast.find_rpp_pole(system, f_hz, zeta0)
    return complex(zeta * w_cm1), float(residual), bool(converged)


def _worker_compute_mode_task(task: Tuple[int, float, float]) -> Tuple[int, complex, float, bool]:
    row_index, w_cm1, phi_offset_deg = task
    if _WORKER_KX_VALUES_CM1 is None:
        raise RuntimeError("Worker kx values not initialized.")
    system = _get_worker_system(phi_offset_deg)
    kx_complex, residual, converged = _find_mode_for_frequency(system, w_cm1, _WORKER_KX_VALUES_CM1, _WORKER_FAST)
    return row_index, kx_complex, residual, converged



def _deserialize_stack(payload: Dict[str, object]) -> StackSpec:
    from .models import DopingSpec

    layers = []
    for l in payload["layers"]:  # type: ignore[index]
        ld = l  # type: ignore[assignment]
        dd = ld["doping"]  # type: ignore[index]
        layers.append(
            LayerSpec(
                material=str(ld["material"]),  # type: ignore[index]
                thickness_m=float(ld["thickness_m"]),  # type: ignore[index]
                euler_deg=tuple(float(v) for v in ld["euler_deg"]),  # type: ignore[index]
                doping=DopingSpec(
                    enabled=bool(dd["enabled"]),  # type: ignore[index]
                    wp_cm1=float(dd["wp_cm1"]),  # type: ignore[index]
                    gp_cm1=float(dd["gp_cm1"]),  # type: ignore[index]
                ),
            )
        )
    return StackSpec.from_layers(layers)


def _run_parallel_rows(
    w_values_cm1: np.ndarray,
    kx_values_cm1: np.ndarray,
    stack_spec: StackSpec,
    phi_offsets_deg: np.ndarray,
    workers: int,
    progress: ProgressCallback,
    progress_label: str,
    custom_materials: Mapping[str, Mapping[str, Any]] | None = None,
    fast: bool = False,
) -> np.ndarray:
    out = np.empty((len(w_values_cm1), len(kx_values_cm1)), dtype=np.complex128)
    payload = _stack_to_worker_payload(stack_spec, custom_materials=custom_materials)
    kx_array = np.asarray(kx_values_cm1, dtype=float)

    if progress:
        progress(0.0, f"{progress_label}: scheduling workers")

    tasks = [(i, float(w), float(phi_off)) for i, (w, phi_off) in enumerate(zip(w_values_cm1, phi_offsets_deg))]
    chunk = max(1, len(tasks) // (workers * 4))
    prepare_hz = _prepare_grid_hz(w_values_cm1)
    pool = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(payload, kx_array, fast, prepare_hz),
    )
    try:
        done = 0
        total = len(tasks)
        for i, row in pool.map(_worker_compute_row_task, tasks, chunksize=chunk):
            out[i, :] = row
            done += 1
            if progress:
                progress(done / total, f"{progress_label}: {done}/{total} rows")
    except (BrokenProcessPool, OSError):
        pool.shutdown(wait=False, cancel_futures=True)
        if progress:
            progress(0.0, f"{progress_label}: parallel unavailable, falling back to serial")
        system_cache: Dict[float, engine.System] = {}
        total = len(tasks)
        for done, (i, w, phi_off) in enumerate(tasks, start=1):
            system = system_cache.get(phi_off)
            if system is None:
                local_stack = stack_spec.with_interior_gamma_offset(phi_off) if phi_off != 0.0 else stack_spec
                system = _build_system(local_stack, custom_materials=payload.get("custom_materials", {}))
                if prepare_hz is not None:
                    system.prepare_frequencies(prepare_hz)
                system_cache[phi_off] = system
            out[i, :] = _compute_row(system, w, kx_array, fast)
            if progress:
                progress(done / total, f"{progress_label}: {done}/{total} rows (serial fallback)")
    except BaseException:
        # UI/session interruptions (disconnect/rerun/stop) should terminate worker tasks promptly.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True, cancel_futures=False)

    return out


def compute_rpp_map(
    stack_spec: StackSpec,
    w_min: float,
    w_max: float,
    nw: int,
    kx_min: float,
    kx_max: float,
    nk: int,
    workers: Optional[int] = None,
    progress: ProgressCallback = None,
    custom_materials: Mapping[str, Mapping[str, Any]] | None = None,
    fast: bool = False,
    return_complex: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Im(rpp) on a regular dispersion grid in (w, kx).

    ``return_complex=True`` returns the complex rpp instead (for real-space field maps).

    When ``fast`` is set the vectorised per-row solver is used; it is numerically
    identical to the default per-point solver but markedly faster on dense grids.
    """
    stack = stack_spec.enforce_boundary_layers()
    stack.validate()

    w_values = np.linspace(w_min, w_max, int(nw), dtype=float)
    kx_values = np.linspace(kx_min, kx_max, int(nk), dtype=float)

    if workers is None:
        workers = _default_workers()
    workers = _resolve_workers(
        workers,
        _estimate_serial_seconds(
            _build_system(stack, custom_materials=custom_materials),
            float(w_values[len(w_values) // 2]) * CM1_TO_HZ, len(w_values), len(kx_values),
        ),
    )

    if workers > 1:
        rpp = _run_parallel_rows(
            w_values,
            kx_values,
            stack,
            phi_offsets_deg=np.zeros_like(w_values),
            workers=workers,
            progress=progress,
            progress_label="Computing Im(rpp)(w,kx)",
            custom_materials=custom_materials,
            fast=fast,
        )
    else:
        system = _build_system(stack, custom_materials=custom_materials)
        prepare_hz = _prepare_grid_hz(w_values)
        if prepare_hz is not None:
            system.prepare_frequencies(prepare_hz)
        rpp = np.empty((len(w_values), len(kx_values)), dtype=np.complex128)
        for i, w in enumerate(w_values):
            rpp[i, :] = _compute_row(system, float(w), kx_values, fast)
            if progress:
                progress((i + 1) / len(w_values), f"Computing Im(rpp)(w,kx): {i + 1}/{len(w_values)} rows")

    _remember_seed(_seed_key(stack, w_min, w_max, nw, kx_min, kx_max, nk, custom_materials), rpp)
    return w_values, kx_values, rpp if return_complex else np.imag(rpp)


def _mirror_symmetric_about_phi0(stack: StackSpec) -> bool:
    """True when ``rpp(-phi) == rpp(phi)`` for the global azimuth sweep of ``stack``.

    The sweep adds ``phi`` to every layer's third Euler angle. Without out-of-plane
    tilt (``beta = 0``) each layer's tensor is then a lab-frame z rotation by
    ``alpha + gamma + phi`` of a tensor that is diagonal in the crystal axes, and a
    y -> -y mirror (which leaves p -> p reflection at k along x unchanged) maps
    ``phi -> -phi`` and ``alpha + gamma -> -(alpha + gamma)``. The sweep is
    therefore symmetric whenever every ``alpha + gamma`` is a multiple of 90 deg.
    """
    for layer in stack.layers:
        alpha, beta, gamma = layer.euler_deg
        if abs(beta) > 1e-9:  # also rejects 360 deg; conservative, only skips a speed-up
            return False
        if abs((alpha + gamma + 45.0) % 90.0 - 45.0) > 1e-9:
            return False
    return True


def _unique_azimuths(phi_offsets_deg: np.ndarray, mirror: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce an azimuth list to the offsets that actually need computing.

    Each layer's tensor is diagonal in its crystal frame, so a rotation of the
    crystal about its own third axis by 180 deg is a symmetry: ``rpp(phi + 180) =
    rpp(phi)`` exactly (for any tilt). With ``mirror`` also ``rpp(-phi) = rpp(phi)``.
    Returns ``(representative_offsets, inverse)`` with
    ``full_rows = rows_of_representatives[inverse]``. A 0-360 deg sweep needs only
    half (or a quarter) of the rows; identical offsets (non-global sweep) collapse to one.
    """
    offsets = np.asarray(phi_offsets_deg, dtype=float)
    canon = np.mod(offsets, 180.0)
    if mirror:
        canon = np.minimum(canon, 180.0 - canon)
    key = np.mod(np.round(canon, 9), 180.0)
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    return offsets[first], inverse.reshape(-1)


def compute_isofreq_map(
    stack_spec: StackSpec,
    w0: float,
    kx_min: float,
    kx_max: float,
    nk: int,
    nphi: int,
    phi_min_deg: float = 0.0,
    phi_max_deg: float = 360.0,
    global_phi_sweep: bool = True,
    workers: Optional[int] = None,
    progress: ProgressCallback = None,
    custom_materials: Mapping[str, Mapping[str, Any]] | None = None,
    fast: bool = False,
    return_complex: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Im(rpp) on a regular isofrequency grid in (phi, kx).

    When ``fast`` is set the vectorised per-row solver is used; it is numerically
    identical to the default per-point solver but markedly faster on dense grids.
    ``return_complex=True`` retains the complex rpp samples for coherent field
    reconstruction, while the default continues to return Im(rpp).
    """
    stack = stack_spec.enforce_boundary_layers()
    stack.validate()

    phi_values_deg = np.linspace(float(phi_min_deg), float(phi_max_deg), int(nphi), endpoint=False, dtype=float)
    phi_values = np.deg2rad(phi_values_deg)
    kx_values = np.linspace(kx_min, kx_max, int(nk), dtype=float)

    if workers is None:
        workers = _default_workers()

    # Only the azimuths that are not related by symmetry are computed; the rest are copies.
    phi_all_deg = phi_values_deg if global_phi_sweep else np.zeros_like(phi_values_deg)
    phi_offsets_deg, inverse = _unique_azimuths(phi_all_deg, _mirror_symmetric_about_phi0(stack))
    w_rows = np.full(phi_offsets_deg.shape[0], float(w0), dtype=float)
    n_rows = phi_offsets_deg.shape[0]
    workers = _resolve_workers(
        workers,
        _estimate_serial_seconds(
            _build_system(stack, custom_materials=custom_materials), float(w0) * CM1_TO_HZ, n_rows, len(kx_values)
        ),
    )

    if workers > 1:
        rpp_unique = _run_parallel_rows(
            w_rows,
            kx_values,
            stack,
            phi_offsets_deg=phi_offsets_deg,
            workers=min(workers, n_rows),
            progress=progress,
            progress_label="Computing isofrequency Im(rpp)(phi,kx)",
            custom_materials=custom_materials,
            fast=fast,
        )
    else:
        rpp_unique = np.empty((n_rows, len(kx_values)), dtype=np.complex128)
        for i, phi_deg in enumerate(phi_offsets_deg):
            local_stack = stack.with_interior_gamma_offset(float(phi_deg)) if global_phi_sweep else stack
            system = _build_system(local_stack, custom_materials=custom_materials)
            rpp_unique[i, :] = _compute_row(system, float(w0), kx_values, fast)
            if progress:
                progress((i + 1) / n_rows, f"Computing isofrequency Im(rpp): {i + 1}/{n_rows} angles")

    rpp = rpp_unique[inverse]

    return phi_values, kx_values, rpp if return_complex else np.imag(rpp)


def compute_isofreq_poles(
    stack_spec: StackSpec,
    w0: float,
    phi_rad: np.ndarray,
    k_cm1: np.ndarray,
    rpp: np.ndarray,
    custom_materials: Mapping[str, Mapping[str, Any]] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complex pole kx(phi) of rpp at fixed frequency, for every azimuth of an isofrequency sweep.

    ``rpp`` is the complex (Nphi, Nk) map from :func:`compute_isofreq_map`; its |rpp| maximum
    on each row seeds a lock-step damped Newton on 1/rpp. Returns ``(k_pole, dinv_dk, ok)``:
    the pole (cm^-1), the slope d(1/rpp)/dk at it (1/cm^-1; the residue is its inverse) and
    a mask of azimuths holding a converged, evanescent, decaying (Im > 0) pole.
    """
    stack = stack_spec.enforce_boundary_layers()
    stack.validate()
    phi_deg = np.rad2deg(np.asarray(phi_rad, dtype=float))
    f = float(w0) * CM1_TO_HZ
    batch = solver_fast.concat_batches([
        _build_system(stack.with_interior_gamma_offset(float(p)), custom_materials=custom_materials).batch_at(np.array([f]))
        for p in phi_deg
    ])
    n = phi_deg.size
    f_all = np.full(n, f)
    seed = np.asarray(k_cm1, dtype=float)[np.argmax(np.abs(rpp), axis=1)] / w0
    zeta, _res, conv = solver_fast.find_rpp_poles_batch(batch, f_all, seed.astype(np.complex128))
    h = 1e-6 * np.maximum(np.abs(zeta), 1.0)
    inv_p = solver_fast._batch_ratio(batch, f_all, zeta + h, True)
    inv_m = solver_fast._batch_ratio(batch, f_all, zeta - h, True)
    dinv = (inv_p - inv_m) / (2.0 * h) / w0
    ok = conv & (zeta.real > 1.0) & (zeta.imag > 0.0) & np.isfinite(dinv) & (dinv != 0)
    return zeta * w0, dinv, ok


def surface_wave_permittivity(
    stack_spec: StackSpec, w0: float,
    custom_materials: Mapping[str, Mapping[str, Any]] | None = None,
) -> complex:
    """Return the isotropic superstrate permittivity for the dipole Green function."""
    layer = _build_layer(stack_spec.layers[0], custom_materials=custom_materials)
    epsilon = layer.calculate_epsilon(float(w0) * CM1_TO_HZ)
    if not layer.is_isotropic:
        raise ValueError("The dipole wave map requires an isotropic superstrate.")
    return complex(epsilon[0, 0])


def _run_parallel_modes(
    w_values_cm1: np.ndarray,
    kx_values_cm1: np.ndarray,
    stack_spec: StackSpec,
    workers: int,
    progress: ProgressCallback,
    custom_materials: Mapping[str, Mapping[str, Any]] | None = None,
    fast: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel per-frequency complex-zero search, mirroring :func:`_run_parallel_rows`."""
    n = len(w_values_cm1)
    kx_mode = np.full(n, np.nan + 1j * np.nan, dtype=np.complex128)
    residual = np.full(n, np.inf, dtype=float)
    converged = np.zeros(n, dtype=bool)

    payload = _stack_to_worker_payload(stack_spec, custom_materials=custom_materials)
    kx_array = np.asarray(kx_values_cm1, dtype=float)

    if progress:
        progress(0.0, "Locating optical modes: scheduling workers")

    tasks = [(i, float(w), 0.0) for i, w in enumerate(w_values_cm1)]
    chunk = max(1, len(tasks) // (workers * 4))
    prepare_hz = _prepare_grid_hz(w_values_cm1)
    pool = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(payload, kx_array, fast, prepare_hz),
    )
    try:
        done = 0
        total = len(tasks)
        for i, kx_c, res, ok in pool.map(_worker_compute_mode_task, tasks, chunksize=chunk):
            kx_mode[i] = kx_c
            residual[i] = res
            converged[i] = ok
            done += 1
            if progress:
                progress(done / total, f"Locating optical modes: {done}/{total} frequencies")
    except (BrokenProcessPool, OSError):
        pool.shutdown(wait=False, cancel_futures=True)
        if progress:
            progress(0.0, "Locating optical modes: parallel unavailable, falling back to serial")
        system = _build_system(stack_spec, custom_materials=payload.get("custom_materials", {}))
        if prepare_hz is not None:
            system.prepare_frequencies(prepare_hz)
        total = len(tasks)
        for done, (i, w, _phi) in enumerate(tasks, start=1):
            kx_mode[i], residual[i], converged[i] = _find_mode_for_frequency(system, w, kx_array, fast)
            if progress:
                progress(done / total, f"Locating optical modes: {done}/{total} frequencies (serial fallback)")
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True, cancel_futures=False)

    return kx_mode, residual, converged


def _worker_compute_mode_chunk(w_chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Worker task: seed grid + lock-step Newton for a contiguous slice of frequencies."""
    if _WORKER_KX_VALUES_CM1 is None:
        raise RuntimeError("Worker kx values not initialized.")
    return _find_modes_vectorised(_get_worker_system(0.0), np.asarray(w_chunk, dtype=float), _WORKER_KX_VALUES_CM1, None)


def _run_parallel_mode_chunks(
    w_values_cm1: np.ndarray,
    kx_values_cm1: np.ndarray,
    stack_spec: StackSpec,
    workers: int,
    progress: ProgressCallback,
    custom_materials: Mapping[str, Mapping[str, Any]] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the frequencies into one contiguous chunk per worker process.

    Every chunk is handled by the vectorised finder, so each process pays its Python
    overhead once per Newton stage for its whole chunk instead of once per frequency.
    """
    n = len(w_values_cm1)
    kx_mode = np.full(n, np.nan + 1j * np.nan, dtype=np.complex128)
    residual = np.full(n, np.inf, dtype=float)
    converged = np.zeros(n, dtype=bool)
    payload = _stack_to_worker_payload(stack_spec, custom_materials=custom_materials)
    kx_array = np.asarray(kx_values_cm1, dtype=float)
    chunks = [c for c in np.array_split(np.arange(n), workers) if c.size]

    if progress:
        progress(0.0, "Locating optical modes: scheduling workers")
    pool = ProcessPoolExecutor(max_workers=len(chunks), initializer=_worker_init, initargs=(payload, kx_array, True, None))
    try:
        futures = [pool.submit(_worker_compute_mode_chunk, np.asarray(w_values_cm1)[c]) for c in chunks]
        for done, (c, fut) in enumerate(zip(chunks, futures), start=1):
            kx_mode[c], residual[c], converged[c] = fut.result()
            if progress:
                progress(done / len(chunks), f"Locating optical modes: {done}/{len(chunks)} blocks")
    except (BrokenProcessPool, OSError):
        pool.shutdown(wait=False, cancel_futures=True)
        if progress:
            progress(0.0, "Locating optical modes: parallel unavailable, falling back to serial")
        system = _build_system(stack_spec, custom_materials=payload.get("custom_materials", {}))
        kx_mode[:], residual[:], converged[:] = _find_modes_vectorised(system, np.asarray(w_values_cm1, dtype=float), kx_array, None)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True, cancel_futures=False)
    return kx_mode, residual, converged


# |rpp| grids of recent dispersion maps. The mode finder seeds from the |rpp| maximum on
# exactly the grid of the displayed dispersion map; remembering that grid lets a
# mode trace on the same stack and grid skip re-evaluating it (the dominant cost for
# tilted anisotropic layers). Keyed by the full set of inputs, so a hit is exact.
_SEED_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_SEED_CACHE_SIZE = 4


def _seed_key(stack: StackSpec, w_min, w_max, nw, kx_min, kx_max, nk, custom_materials) -> tuple | None:
    try:
        custom = json.dumps(custom_materials or {}, sort_keys=True, default=str)
        return (stack, float(w_min), float(w_max), int(nw), float(kx_min), float(kx_max), int(nk), custom)
    except (TypeError, ValueError):
        return None


def _remember_seed(key: tuple | None, rpp: np.ndarray) -> None:
    if key is None:
        return
    _SEED_CACHE[key] = np.abs(rpp)
    _SEED_CACHE.move_to_end(key)
    while len(_SEED_CACHE) > _SEED_CACHE_SIZE:
        _SEED_CACHE.popitem(last=False)


def _find_modes_vectorised(
    system: engine.System,
    w_values: np.ndarray,
    kx_values: np.ndarray,
    seed_abs: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All frequencies at once: |rpp|-maximum seeds, then lock-step damped Newton on 1/rpp."""
    f_hz = w_values * CM1_TO_HZ
    if seed_abs is None:
        seed_abs = np.abs(solver_fast.rpp_grid(system, w_values, kx_values))
    zeta0 = kx_values[np.argmax(seed_abs, axis=1)] / w_values
    zeta, residual, converged = solver_fast.find_rpp_poles_batch(system.batch_at(f_hz), f_hz, zeta0.astype(np.complex128))
    zeta, residual, converged = _repair_mode_trace(system, f_hz, w_values, kx_values, zeta, residual, converged)
    return zeta * w_values, residual, converged


def _repair_kx_trace(system, w_values, kx_values, kx_mode, residual, converged):
    """:func:`_repair_mode_trace` for results expressed as kx (cm⁻¹) instead of zeta."""
    zeta, residual, converged = _repair_mode_trace(
        system, w_values * CM1_TO_HZ, w_values, kx_values, kx_mode / w_values, residual, converged
    )
    return zeta * w_values, residual, converged


def _repair_mode_trace(
    system: engine.System,
    f_hz: np.ndarray,
    w_values: np.ndarray,
    kx_values: np.ndarray,
    zeta: np.ndarray,
    residual: np.ndarray,
    converged: np.ndarray,
    *,
    window: int = 3,
    passes: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove isolated jumps from a mode trace and re-seed failed points from their neighbours.

    The |rpp| maximum of one frequency row can sit on another branch or on a non-resonant
    edge peak, so single frequencies land on a pole far from the trace. Points whose Re(kx)
    is outside the map range, or further than ``tol`` from the median of their neighbours
    (``window`` samples each side), are dropped; they and the failed points close to the
    trace are then re-run from the neighbour-interpolated pole. A re-run is kept only if it
    converges inside the map range and within ``tol`` of that interpolation.
    """
    lo, hi = float(kx_values[0]), float(kx_values[-1])
    span = max(hi - lo, 1e-30)
    zeta, residual, converged = zeta.copy(), residual.copy(), converged.copy()
    n = len(w_values)
    index = np.arange(n)

    def in_range(z: np.ndarray) -> np.ndarray:
        kx = z.real * w_values
        return np.isfinite(kx) & (kx >= lo) & (kx <= hi)

    converged &= in_range(zeta)
    for _ in range(passes):
        kx = np.where(converged, zeta.real * w_values, np.nan)
        good_idx = np.flatnonzero(converged)
        if good_idx.size < 3:
            break
        step = np.abs(np.diff(kx[good_idx]))
        tol = max(4.0 * float(np.median(step)), 0.03 * span)
        offsets = [d for d in range(-window, window + 1) if d]
        neigh = np.full((n, len(offsets)), np.nan)
        for c, d in enumerate(offsets):
            src = index + d
            valid = (src >= 0) & (src < n)
            neigh[valid, c] = kx[src[valid]]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = np.nanmedian(neigh, axis=1)
            count = np.sum(np.isfinite(neigh), axis=1)
        jump = converged & (count >= 2) & (np.abs(kx - ref) > tol)
        keep = converged & ~jump
        if keep.sum() < 2:
            converged = keep
            break
        # candidates: everything not kept that lies within `window` samples of a kept point
        gap = np.min(np.abs(index[:, None] - np.flatnonzero(keep)[None, :]), axis=1)
        todo = np.flatnonzero(~keep & (gap <= window))
        converged = keep
        if todo.size == 0:
            break
        k_idx = np.flatnonzero(keep)
        seed = np.interp(todo, k_idx, zeta[k_idx].real) + 1j * np.interp(todo, k_idx, zeta[k_idx].imag)
        z_new, r_new, c_new = solver_fast.find_rpp_poles_batch(system.batch_at(f_hz[todo]), f_hz[todo], seed)
        kx_new = z_new.real * w_values[todo]
        kx_seed = seed.real * w_values[todo]
        accept = c_new & np.isfinite(kx_new) & (kx_new >= lo) & (kx_new <= hi) & (np.abs(kx_new - kx_seed) <= tol)
        zeta[todo[accept]], residual[todo[accept]] = z_new[accept], r_new[accept]
        converged[todo[accept]] = True
        if not accept.any():
            break
    return zeta, residual, converged


def compute_mode_dispersion(
    stack_spec: StackSpec,
    w_min: float,
    w_max: float,
    nw: int,
    kx_min: float,
    kx_max: float,
    nk: int,
    workers: Optional[int] = None,
    progress: ProgressCallback = None,
    custom_materials: Mapping[str, Mapping[str, Any]] | None = None,
    fast: bool = False,
    seed_abs_rpp: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trace the optical mode (complex pole of rpp) as a function of frequency.

    For each frequency the reduced wavevector ``zeta`` of the mode -- the pole of
    rpp, where the p-polarised dispersion determinant vanishes -- is found: the
    ``|rpp|`` maximum on the real ``kx`` grid seeds a complex-plane Newton
    refinement of ``1/rpp = 0``. The returned ``kx_mode`` is complex
    (``zeta * w`` in cm⁻¹); its real part is the mode wavevector and its imaginary
    part the modal loss.

    With ``fast`` all frequencies are processed together: the seed grid is one set of
    large batched evaluations and the Newton iterations of every frequency advance in
    lock-step (same per-frequency iteration as the scalar finder, ~3-5x faster, ~1.5x
    for tilted anisotropic layers). ``|rpp|`` of the same stack and grid, if a dispersion
    map was just computed with :func:`compute_rpp_map` (or passed as ``seed_abs_rpp``,
    shape ``(nw, nk)``), is reused instead of re-evaluated. Without a reusable seed a
    job large enough to pay for it is split into one block of frequencies per worker
    process (each block runs the vectorised finder).

    Returns ``(w_values, kx_mode_cm1, residual, converged)``. ``residual`` is the
    final ``|1/rpp|`` and ``converged`` flags the frequencies where the search
    reached a genuine pole (callers should drop the rest).
    """
    stack = stack_spec.enforce_boundary_layers()
    stack.validate()

    w_values = np.linspace(w_min, w_max, int(nw), dtype=float)
    kx_values = np.linspace(kx_min, kx_max, int(nk), dtype=float)

    if workers is None:
        workers = _default_workers()

    if fast:
        system = _build_system(stack, custom_materials=custom_materials)
        if seed_abs_rpp is None:
            seed_abs_rpp = _SEED_CACHE.get(_seed_key(stack, w_min, w_max, nw, kx_min, kx_max, nk, custom_materials))
        elif np.shape(seed_abs_rpp) != (len(w_values), len(kx_values)):
            raise ValueError("seed_abs_rpp must have shape (nw, nk).")
        if seed_abs_rpp is None and workers > 1:
            # Without a reusable seed grid the job is the grid plus the Newton stages.
            # Large jobs go to a process pool, one block of frequencies per worker.
            estimate = _estimate_serial_seconds(
                system, float(w_values[len(w_values) // 2]) * CM1_TO_HZ, len(w_values), len(kx_values),
                extra_us_per_row=700.0,
            )
            if _resolve_workers(workers, estimate) > 1 and len(w_values) >= 2 * workers:
                kx_mode, residual, converged = _run_parallel_mode_chunks(
                    w_values, kx_values, stack, workers, progress, custom_materials
                )
                return (w_values, *_repair_kx_trace(system, w_values, kx_values, kx_mode, residual, converged))
        if progress:
            progress(0.0, "Locating optical modes")
        kx_mode, residual, converged = _find_modes_vectorised(
            system, w_values, kx_values, None if seed_abs_rpp is None else np.asarray(seed_abs_rpp, dtype=float)
        )
        if progress:
            progress(1.0, "Locating optical modes: done")
        return w_values, kx_mode, residual, converged

    workers = _resolve_workers(
        workers,
        _estimate_serial_seconds(
            _build_system(stack, custom_materials=custom_materials),
            float(w_values[len(w_values) // 2]) * CM1_TO_HZ, len(w_values), len(kx_values),
            extra_us_per_row=2500.0,  # complex-plane Newton refinement per frequency
        ),
    )

    if workers > 1:
        kx_mode, residual, converged = _run_parallel_modes(
            w_values,
            kx_values,
            stack,
            workers=workers,
            progress=progress,
            custom_materials=custom_materials,
            fast=fast,
        )
    else:
        kx_mode = np.full(len(w_values), np.nan + 1j * np.nan, dtype=np.complex128)
        residual = np.full(len(w_values), np.inf, dtype=float)
        converged = np.zeros(len(w_values), dtype=bool)
        system = _build_system(stack, custom_materials=custom_materials)
        prepare_hz = _prepare_grid_hz(w_values)
        if prepare_hz is not None:
            system.prepare_frequencies(prepare_hz)
        for i, w in enumerate(w_values):
            kx_mode[i], residual[i], converged[i] = _find_mode_for_frequency(system, float(w), kx_values, fast)
            if progress:
                progress((i + 1) / len(w_values), f"Locating optical modes: {i + 1}/{len(w_values)} frequencies")

    return w_values, kx_mode, residual, converged
