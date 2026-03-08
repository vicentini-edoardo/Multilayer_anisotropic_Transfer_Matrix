"""Numerical solver routines for dispersion and isofrequency Im(rpp) maps."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import os
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .materials import CM1_TO_HZ, axes_for_material
from .models import LayerSpec, StackSpec

try:
    import GTM.GTMcore as GTM  # type: ignore
except Exception as exc:  # pragma: no cover - exercised by runtime environment setup
    raise ImportError(
        "pyGTM is required but not importable. Install a compatible pyGTM package "
        "(for example via `pip install \"pyGTM @ git+https://github.com/pyMatJ/pyGTM.git@7a228b7314ea66ae025ff346c0d0e8bfb86cc82c\"`) and retry."
    ) from exc


ProgressCallback = Optional[Callable[[float, str], None]]

DELTA1234 = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.complex128,
)

_WORKER_STACK_PAYLOAD: Optional[Dict[str, object]] = None
_WORKER_KX_VALUES_CM1: Optional[np.ndarray] = None
_WORKER_SYSTEM_CACHE: Dict[float, GTM.System] = {}


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
    """
    alpha, beta, gamma = euler_deg
    theta = np.deg2rad(beta)
    phi = np.deg2rad(alpha)
    psi = np.deg2rad(gamma)
    return float(theta), float(phi), float(psi)


def _default_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(4, cpu))


def _build_system(stack_spec: StackSpec) -> GTM.System:
    stack = stack_spec.enforce_boundary_layers()
    system = GTM.System()

    layers = list(stack.layers)
    super_layer = _build_layer(layers[0])
    sub_layer = _build_layer(layers[-1])
    system.set_superstrate(super_layer)
    system.set_substrate(sub_layer)

    for mid in layers[1:-1]:
        system.add_layer(_build_layer(mid))

    return system


def _build_layer(layer_spec: LayerSpec) -> GTM.Layer:
    axes = axes_for_material(layer_spec.material, layer_spec.doping)
    theta, phi, psi = passler_to_pygtm_euler(layer_spec.euler_deg)
    return GTM.Layer(
        thickness=float(layer_spec.thickness_m),
        epsilon1=axes.fx,
        epsilon2=axes.fy,
        epsilon3=axes.fz,
        theta=theta,
        phi=phi,
        psi=psi,
    )


def _update_layer_no_eps(layer: GTM.Layer, f_hz: float, zeta: complex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    layer.calculate_matrices(zeta)
    layer.calculate_q()
    layer.calculate_gamma(zeta)
    layer.calculate_transfer_matrix(f_hz, zeta)
    ai = layer.Ai
    ai_inv = GTM.exact_inv(ai)
    return ai, ai_inv, layer.Ti


def _calculate_gamma_star_no_eps(system: GTM.System, f_hz: float, zeta: complex) -> np.ndarray:
    _, ai_inv_super, _ = _update_layer_no_eps(system.superstrate, f_hz, zeta)
    ai_sub, _, _ = _update_layer_no_eps(system.substrate, f_hz, zeta)

    tloc = np.identity(4, dtype=np.complex128)
    for ii in range(len(system.layers) - 1, -1, -1):
        _, _, ti = _update_layer_no_eps(system.layers[ii], f_hz, zeta)
        tloc = ti @ tloc

    gamma = ai_inv_super @ (tloc @ ai_sub)
    gamma_star = DELTA1234 @ (gamma @ DELTA1234)
    return gamma_star


def _rpp_from_gamma_star(gamma_star: np.ndarray) -> complex:
    denom = gamma_star[0, 0] * gamma_star[2, 2] - gamma_star[0, 2] * gamma_star[2, 0]
    if denom == 0 or not np.isfinite(denom):
        return 0.0 + 0.0j
    rpp = gamma_star[1, 0] * gamma_star[2, 2] - gamma_star[1, 2] * gamma_star[2, 0]
    out = rpp / denom
    if not np.isfinite(out):
        return 0.0 + 0.0j
    return out


def _compute_row_with_system(system: GTM.System, w_cm1: float, kx_cm1: np.ndarray) -> np.ndarray:
    f_hz = float(cm1_to_hz(w_cm1))
    system.initialize_sys(f_hz)
    row = np.empty(kx_cm1.shape[0], dtype=np.complex128)
    zeta = np.asarray(kx_cm1, dtype=float) * (1.0 / float(w_cm1))
    for j, z in enumerate(zeta):
        # Small imaginary regularization avoids pyGTM mode-sorting singular branches.
        z_reg = complex(float(np.real(z)), 1e-14)
        try:
            gamma_star = _calculate_gamma_star_no_eps(system, f_hz, z_reg)
        except Exception:
            try:
                gamma_star = system.calculate_GammaStar(f_hz, z_reg)
            except Exception:
                row[j] = 0.0 + 0.0j
                continue
        row[j] = _rpp_from_gamma_star(gamma_star)
    return row


def _stack_to_worker_payload(stack_spec: StackSpec) -> Dict[str, object]:
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
        ]
    }


def _worker_init(stack_payload: Dict[str, object], kx_values_cm1: np.ndarray) -> None:
    global _WORKER_STACK_PAYLOAD, _WORKER_KX_VALUES_CM1, _WORKER_SYSTEM_CACHE
    _WORKER_STACK_PAYLOAD = stack_payload
    _WORKER_KX_VALUES_CM1 = np.asarray(kx_values_cm1, dtype=float)
    _WORKER_SYSTEM_CACHE = {}


def _get_worker_system(phi_offset_deg: float) -> GTM.System:
    if _WORKER_STACK_PAYLOAD is None:
        raise RuntimeError("Worker stack payload not initialized.")

    key = float(phi_offset_deg)
    cached = _WORKER_SYSTEM_CACHE.get(key)
    if cached is not None:
        return cached

    stack = _deserialize_stack(_WORKER_STACK_PAYLOAD)
    if key != 0.0:
        stack = stack.with_interior_alpha_offset(key)
    system = _build_system(stack)
    _WORKER_SYSTEM_CACHE[key] = system
    return system


def _worker_compute_row_task(task: Tuple[int, float, float]) -> Tuple[int, np.ndarray]:
    row_index, w_cm1, phi_offset_deg = task
    if _WORKER_KX_VALUES_CM1 is None:
        raise RuntimeError("Worker kx values not initialized.")
    system = _get_worker_system(phi_offset_deg)
    row = _compute_row_with_system(system, w_cm1, _WORKER_KX_VALUES_CM1)
    return row_index, row



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
) -> np.ndarray:
    out = np.empty((len(w_values_cm1), len(kx_values_cm1)), dtype=np.complex128)
    payload = _stack_to_worker_payload(stack_spec)
    kx_array = np.asarray(kx_values_cm1, dtype=float)

    if progress:
        progress(0.0, f"{progress_label}: scheduling workers")

    tasks = [(i, float(w), float(phi_off)) for i, (w, phi_off) in enumerate(zip(w_values_cm1, phi_offsets_deg))]
    chunk = max(1, len(tasks) // (workers * 4))
    pool = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(payload, kx_array),
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
        system_cache: Dict[float, GTM.System] = {}
        total = len(tasks)
        for done, (i, w, phi_off) in enumerate(tasks, start=1):
            system = system_cache.get(phi_off)
            if system is None:
                local_stack = stack_spec.with_interior_alpha_offset(phi_off) if phi_off != 0.0 else stack_spec
                system = _build_system(local_stack)
                system_cache[phi_off] = system
            out[i, :] = _compute_row_with_system(system, w, kx_array)
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Im(rpp) on a regular dispersion grid in (w, kx)."""
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
        )
    else:
        system = _build_system(stack)
        rpp = np.empty((len(w_values), len(kx_values)), dtype=np.complex128)
        for i, w in enumerate(w_values):
            rpp[i, :] = _compute_row_with_system(system, float(w), kx_values)
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
    global_phi_sweep: bool = True,
    workers: Optional[int] = None,
    progress: ProgressCallback = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Im(rpp) on a regular isofrequency grid in (phi, kx)."""
    stack = stack_spec.enforce_boundary_layers()
    stack.validate()

    phi_values = np.linspace(0.0, 2.0 * np.pi, int(nphi), endpoint=False, dtype=float)
    kx_values = np.linspace(kx_min, kx_max, int(nk), dtype=float)

    if workers is None:
        workers = _default_workers()

    w_rows = np.full(phi_values.shape[0], float(w0), dtype=float)
    phi_offsets_deg = np.rad2deg(phi_values) if global_phi_sweep else np.zeros_like(phi_values)

    if workers > 1:
        rpp = _run_parallel_rows(
            w_rows,
            kx_values,
            stack,
            phi_offsets_deg=phi_offsets_deg,
            workers=workers,
            progress=progress,
            progress_label="Computing isofrequency Im(rpp)(phi,kx)",
        )
    else:
        rpp = np.empty((len(phi_values), len(kx_values)), dtype=np.complex128)
        for i, phi_deg in enumerate(phi_offsets_deg):
            local_stack = stack.with_interior_alpha_offset(float(phi_deg)) if global_phi_sweep else stack
            system = _build_system(local_stack)
            rpp[i, :] = _compute_row_with_system(system, float(w0), kx_values)
            if progress:
                progress((i + 1) / len(phi_values), f"Computing isofrequency Im(rpp): {i + 1}/{len(phi_values)} angles")

    return phi_values, kx_values, np.imag(rpp)
