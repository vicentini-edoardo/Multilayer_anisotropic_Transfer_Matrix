"""Numerical solver routines for dispersion and isofrequency Im(rpp) maps."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

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


def cm1_to_hz(values_cm1: np.ndarray | float) -> np.ndarray:
    """Convert wavenumbers in cm⁻¹ to frequencies in Hz."""
    return np.asarray(values_cm1, dtype=float) * CM1_TO_HZ


def zeta_from_kx_w(kx_cm1: np.ndarray | float, w_cm1: np.ndarray | float) -> np.ndarray:
    """Convert in-plane momentum and frequency to the reduced wavevector zeta."""
    w = np.asarray(w_cm1, dtype=float)
    kx = np.asarray(kx_cm1, dtype=float)
    return kx / w


def passler_to_pygtm_euler(euler_deg: Sequence[float]) -> Tuple[float, float, float]:
    """
    Convert MATLAB Passler z-x-z Euler convention to pyGTM ordering.

    Passler angles are (alpha, beta, gamma) in degrees.
    pyGTM expects arguments (theta, phi, psi) where:
    theta = beta, phi = alpha, psi = gamma.

    NOTE: this is an angle re-labelling only. It is exact when both libraries
    share the same intrinsic z-x-z axis sequence. If pyGTM is ever found to use
    a different sequence (e.g. z-y-z), this mapping must be corrected and
    validated against a known reference (e.g. a uniaxial crystal at a fixed
    orientation) — relabelling alone would otherwise introduce a silent
    physics error.
    """
    alpha, beta, gamma = euler_deg
    theta = np.deg2rad(beta)
    phi = np.deg2rad(alpha)
    psi = np.deg2rad(gamma)
    return float(theta), float(phi), float(psi)


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
    theta, phi, psi = passler_to_pygtm_euler(layer_spec.euler_deg)
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


def _worker_init(stack_payload: Dict[str, object], kx_values_cm1: np.ndarray, fast: bool) -> None:
    global _WORKER_STACK_PAYLOAD, _WORKER_KX_VALUES_CM1, _WORKER_SYSTEM_CACHE, _WORKER_FAST
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
    """Locate the complex kx where rpp = 0 at a single frequency.

    The real-kx row gives the starting guess (the kx of minimum |rpp|), which is
    then refined into the complex plane by :func:`solver_fast.find_rpp_zero`.
    Returns ``(kx_complex_cm1, residual, converged)``; ``kx_complex_cm1`` is
    ``zeta * w`` in cm⁻¹ (real part = mode wavevector, imaginary part = loss).
    """
    f_hz = float(w_cm1 * CM1_TO_HZ)
    # Evaluates the row (which initialises the system at f_hz) and gives the
    # coarse grid guess for the zero.
    row = _compute_row(system, w_cm1, kx_values_cm1, fast)
    mag = np.abs(np.asarray(row))
    finite = np.isfinite(mag)
    if not np.any(finite):
        return complex(np.nan, np.nan), float("inf"), False
    j0 = int(np.argmin(np.where(finite, mag, np.inf)))
    zeta0 = complex(float(kx_values_cm1[j0]) / float(w_cm1), 0.0)
    zeta, residual, converged = solver_fast.find_rpp_zero(system, f_hz, zeta0)
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
    pool = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(payload, kx_array, fast),
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
                local_stack = stack_spec.with_interior_alpha_offset(phi_off) if phi_off != 0.0 else stack_spec
                system = _build_system(local_stack, custom_materials=payload.get("custom_materials", {}))
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Im(rpp) on a regular dispersion grid in (w, kx).

    When ``fast`` is set the vectorised per-row solver is used; it is numerically
    identical to the default per-point solver but markedly faster on dense grids.
    """
    stack = stack_spec.enforce_boundary_layers()
    stack.validate()

    w_values = np.linspace(w_min, w_max, int(nw), dtype=float)
    kx_values = np.linspace(kx_min, kx_max, int(nk), dtype=float)

    if workers is None:
        workers = _default_workers()

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
        rpp = np.empty((len(w_values), len(kx_values)), dtype=np.complex128)
        for i, w in enumerate(w_values):
            rpp[i, :] = _compute_row(system, float(w), kx_values, fast)
            if progress:
                progress((i + 1) / len(w_values), f"Computing Im(rpp)(w,kx): {i + 1}/{len(w_values)} rows")

    return w_values, kx_values, np.imag(rpp)


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Im(rpp) on a regular isofrequency grid in (phi, kx).

    When ``fast`` is set the vectorised per-row solver is used; it is numerically
    identical to the default per-point solver but markedly faster on dense grids.
    """
    stack = stack_spec.enforce_boundary_layers()
    stack.validate()

    phi_values_deg = np.linspace(float(phi_min_deg), float(phi_max_deg), int(nphi), endpoint=False, dtype=float)
    phi_values = np.deg2rad(phi_values_deg)
    kx_values = np.linspace(kx_min, kx_max, int(nk), dtype=float)

    if workers is None:
        workers = _default_workers()

    w_rows = np.full(phi_values.shape[0], float(w0), dtype=float)
    phi_offsets_deg = phi_values_deg if global_phi_sweep else np.zeros_like(phi_values_deg)

    if workers > 1:
        rpp = _run_parallel_rows(
            w_rows,
            kx_values,
            stack,
            phi_offsets_deg=phi_offsets_deg,
            workers=workers,
            progress=progress,
            progress_label="Computing isofrequency Im(rpp)(phi,kx)",
            custom_materials=custom_materials,
            fast=fast,
        )
    else:
        rpp = np.empty((len(phi_values), len(kx_values)), dtype=np.complex128)
        for i, phi_deg in enumerate(phi_offsets_deg):
            local_stack = stack.with_interior_gamma_offset(float(phi_deg)) if global_phi_sweep else stack
            system = _build_system(local_stack, custom_materials=custom_materials)
            rpp[i, :] = _compute_row(system, float(w0), kx_values, fast)
            if progress:
                progress((i + 1) / len(phi_values), f"Computing isofrequency Im(rpp): {i + 1}/{len(phi_values)} angles")

    return phi_values, kx_values, np.imag(rpp)


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
        progress(0.0, "Locating reflectivity zeros: scheduling workers")

    tasks = [(i, float(w), 0.0) for i, w in enumerate(w_values_cm1)]
    chunk = max(1, len(tasks) // (workers * 4))
    pool = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(payload, kx_array, fast),
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
                progress(done / total, f"Locating reflectivity zeros: {done}/{total} frequencies")
    except (BrokenProcessPool, OSError):
        pool.shutdown(wait=False, cancel_futures=True)
        if progress:
            progress(0.0, "Locating reflectivity zeros: parallel unavailable, falling back to serial")
        system = _build_system(stack_spec, custom_materials=payload.get("custom_materials", {}))
        total = len(tasks)
        for done, (i, w, _phi) in enumerate(tasks, start=1):
            kx_mode[i], residual[i], converged[i] = _find_mode_for_frequency(system, w, kx_array, fast)
            if progress:
                progress(done / total, f"Locating reflectivity zeros: {done}/{total} frequencies (serial fallback)")
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True, cancel_futures=False)

    return kx_mode, residual, converged


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trace the optical mode (complex zero of rpp) as a function of frequency.

    For each frequency the reduced wavevector ``zeta`` that solves ``rpp(zeta) = 0``
    is found: the ``|rpp|`` minimum on the real ``kx`` grid seeds a complex-plane
    Newton refinement. The returned ``kx_mode`` is complex (``zeta * w`` in cm⁻¹);
    its real part is the mode wavevector and its imaginary part the modal loss.

    Returns ``(w_values, kx_mode_cm1, residual, converged)``. ``residual`` is the
    final ``|rpp|`` and ``converged`` flags the frequencies where the search reached
    a genuine zero (callers should drop the rest).
    """
    stack = stack_spec.enforce_boundary_layers()
    stack.validate()

    w_values = np.linspace(w_min, w_max, int(nw), dtype=float)
    kx_values = np.linspace(kx_min, kx_max, int(nk), dtype=float)

    if workers is None:
        workers = _default_workers()

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
        for i, w in enumerate(w_values):
            kx_mode[i], residual[i], converged[i] = _find_mode_for_frequency(system, float(w), kx_values, fast)
            if progress:
                progress((i + 1) / len(w_values), f"Locating reflectivity zeros: {i + 1}/{len(w_values)} frequencies")

    return w_values, kx_mode, residual, converged
