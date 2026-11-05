"""Calculation setup, result rendering, and export utilities for the Streamlit UI."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO, StringIO
import re
from typing import Any, Callable, Dict, List, Mapping, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..custom_materials import material_frequency_range_warnings
from ..plotting import (
    MARKER_LINE_COLOR,
    plot_heatmap,
    plot_heatmap_interactive,
    plot_polar_isofrequency,
    plot_polar_isofrequency_interactive,
)
from ..solver import compute_isofreq_map, compute_isofreq_poles, compute_mode_dispersion, compute_rpp_map, surface_wave_permittivity
from ..surface_wave import (
    compute_dispersion_wave_ft,
    compute_dispersion_wave_poles,
    compute_pole_wave,
    compute_surface_wave,
)
from .layer_builder import build_stack_from_session
from .material_builder import custom_material_registry
from ..models import StackSpec


MAX_MAP_GRID_POINTS = 250_000
MAX_ISO_GRID_POINTS = 250_000
KX_UI_SCALE = 1e3
FAST_PREVIEW_MAX_X = 220
FAST_PREVIEW_MAX_Y = 220
FAST_PREVIEW_MAX_R = 220
FAST_PREVIEW_MAX_PHI = 220
ISO_DEFAULT_W0 = 900.0
ISO_DEFAULT_KX_MIN = 0.5
ISO_DEFAULT_KX_MAX = 25.0
# Real-space view: Re(Ez) shows the polariton wavefronts; the intensity only shows the decay.
WAVE_METHODS = ("Fourier (all evanescent waves)", "Poles (mode only)")
WAVE_DISPLAYS = ("Re(Ez) wavefronts", "|Ez|² (dB)", "|Ez|² (linear)")
PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}
PLOT_COLORMAPS = {
    "Magma": "Magma",
    "Cividis": "Cividis",
    "Viridis": "Viridis",
    "Plasma": "Plasma",
}

ComputeResultT = TypeVar("ComputeResultT")

CALC_UI_STATE_KEYS = (
    "calc_mode",
    "map_resolution_choice",
    "iso_resolution_choice",
    "worker_count",
    "fast_preview_plots",
    "plot_colormap",
    "plot_manual_color_limits",
    "plot_color_min",
    "plot_color_max",
    "map_state",
    "iso_state",
    "wave_source_height_nm",
    "wave_observation_height_nm",
    "wave_half_width_um",
    "wave_fft_size",
    "wave_display",
    "wave_method",
)

def _initial_calc_defaults(speed_presets: Mapping[str, Mapping[str, Mapping[str, int]]]) -> Dict[str, Any]:
    default_preset = "Normal" if "Normal" in speed_presets else next(iter(speed_presets))
    return {
        "calc_mode": "Im(rpp) as f(w, kx)",
        "map_resolution_choice": default_preset,
        "iso_resolution_choice": default_preset,
        "worker_count": 1,
        "fast_preview_plots": True,
        "plot_colormap": "Magma",
        "plot_manual_color_limits": False,
        "plot_color_min": 0.0,
        "plot_color_max": 1.0,
        "wave_source_height_nm": 25.0,
        "wave_observation_height_nm": 0.0,
        "wave_half_width_um": 2.0,
        "wave_fft_size": 1024,
        "wave_display": WAVE_DISPLAYS[0],
        "wave_method": WAVE_METHODS[0],
        "plot_refresh_nonce": 0,
        "compute_state": "Idle",
        "last_compute_signature": None,
        "last_compute_mode": None,
        "last_compute_timestamp": None,
        "map_result": None,
        "iso_result": None,
        "map_history": [],
        "iso_history": [],
        "selected_map_history_id": None,
        "selected_iso_history_id": None,
        "history_seq": 0,
        "calc_settings": {},
        "map_state": {
            "phi0": 0.0,
            "w_min": 780.0,
            "w_max": 1050.0,
            "nw": int(speed_presets[default_preset]["map"]["nw"]),
            "kx_min": 0.05,
            "kx_max": 15.0,
            "nk": int(speed_presets[default_preset]["map"]["nk"]),
        },
        "iso_state": {
            "w0": ISO_DEFAULT_W0,
            "nk": int(speed_presets[default_preset]["iso"]["nk"]),
            "nphi": int(speed_presets[default_preset]["iso"]["nphi"]),
            "kx_min": ISO_DEFAULT_KX_MIN,
            "kx_max": ISO_DEFAULT_KX_MAX,
            "phi_min_deg": 0.0,
            "phi_max_deg": 360.0,
        },
    }


def _calc_ui_state_snapshot() -> Dict[str, Any]:
    return {key: st.session_state[key] for key in CALC_UI_STATE_KEYS if key in st.session_state}


_LEGACY_KEY_MAP: dict[str, dict[str, str]] = {
    "map_state": {
        "phi0": "map_phi0",
        "w_min": "map_w_min",
        "w_max": "map_w_max",
        "nw": "map_nw",
        "kx_min": "map_kx_min",
        "kx_max": "map_kx_max",
        "nk": "map_nk",
    },
    "iso_state": {
        "w0": "iso_w0",
        "nk": "iso_nk",
        "nphi": "iso_nphi",
        "kx_min": "iso_kx_min",
        "kx_max": "iso_kx_max",
        "phi_min_deg": "iso_phi_min_deg",
        "phi_max_deg": "iso_phi_max_deg",
    },
}


def init_calculation_state(speed_presets: Mapping[str, Mapping[str, Mapping[str, int]]]) -> None:
    """Initialize persistent calculation settings and restore them across reruns."""
    defaults = _initial_calc_defaults(speed_presets)
    preserved_state = st.session_state.pop("_preserved_calc_state", None) or {}
    restored_state = preserved_state

    for key, default_value in defaults.items():
        if key in {"map_state", "iso_state"}:
            bucket = dict(restored_state.get(key, st.session_state.get(key, {})) or {})
            legacy_key_map = _LEGACY_KEY_MAP[key]
            for field, legacy_key in legacy_key_map.items():
                if field not in bucket and legacy_key in st.session_state:
                    bucket[field] = st.session_state[legacy_key]
            merged = dict(default_value)
            merged.update(bucket)
            st.session_state[key] = merged
        else:
            if key not in st.session_state:
                st.session_state[key] = restored_state.get(key, default_value)

    valid_resolution_choices = set(speed_presets) | {"Custom"}
    if st.session_state.get("map_resolution_choice") not in valid_resolution_choices:
        st.session_state.map_resolution_choice = str(defaults["map_resolution_choice"])
    if st.session_state.get("iso_resolution_choice") not in valid_resolution_choices:
        st.session_state.iso_resolution_choice = str(defaults["iso_resolution_choice"])
    if st.session_state.get("calc_mode") not in ("Im(rpp) as f(w, kx)", "Isofrequency surface"):
        st.session_state.calc_mode = str(defaults["calc_mode"])

    if float(st.session_state["iso_state"].get("w0", 0.0)) <= 0.0:
        st.session_state["iso_state"]["w0"] = ISO_DEFAULT_W0
    if float(st.session_state["iso_state"].get("kx_min", 0.0)) <= 0.0:
        st.session_state["iso_state"]["kx_min"] = ISO_DEFAULT_KX_MIN
    if float(st.session_state["iso_state"].get("kx_max", 0.0)) <= float(st.session_state["iso_state"]["kx_min"]):
        st.session_state["iso_state"]["kx_max"] = ISO_DEFAULT_KX_MAX
    if float(st.session_state["iso_state"].get("phi_max_deg", 360.0)) <= float(st.session_state["iso_state"].get("phi_min_deg", 0.0)):
        st.session_state["iso_state"]["phi_min_deg"] = 0.0
        st.session_state["iso_state"]["phi_max_deg"] = 360.0
    _sync_resolution_defaults(speed_presets)


def _stack_role(index: int, total: int) -> str:
    if index == 0:
        return "superstrate"
    if index == total - 1:
        return "substrate"
    return "interior"


def _build_export_txt(
    mode: str,
    stack: StackSpec,
    data: np.ndarray,
    columns: List[str],
    workers: int,
    resolution_name: str,
) -> bytes:
    buffer = StringIO()
    now = datetime.now(timezone.utc).isoformat()
    buffer.write(f"# exported_utc: {now}\n")
    buffer.write(f"# mode: {mode}\n")
    buffer.write(f"# resolution: {resolution_name}\n")
    buffer.write(f"# workers: {workers}\n")
    buffer.write("# units: rpp=Im(rpp), kx=cm^-1, w=cm^-1, phi=rad\n")
    buffer.write(f"# rows: {data.shape[0]}\n")
    buffer.write(f"# columns: {','.join(columns)}\n")
    buffer.write("# layers:\n")
    buffer.write("# layer_index,role,material,thickness_m,thickness_nm,alpha_deg,beta_deg,gamma_deg\n")
    total_layers = len(stack.layers)
    for idx, layer in enumerate(stack.layers):
        alpha, beta, gamma = layer.euler_deg
        role = _stack_role(idx, total_layers)
        thickness_nm = float(layer.thickness_m) * 1e9
        buffer.write(
            f"# {idx},{role},{layer.material},{layer.thickness_m:.12g},{thickness_nm:.12g},"
            f"{alpha:.12g},{beta:.12g},{gamma:.12g}\n"
        )
    buffer.write(",".join(columns) + "\n")
    np.savetxt(buffer, data, delimiter=",", fmt="%.12g")
    return buffer.getvalue().encode("utf-8")


def _slug_token(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned.lower() or "layer"


def _stack_plot_filename_stem(mode: str, stack: StackSpec | None = None) -> str:
    stack = stack if stack is not None else build_stack_from_session()
    layer_tokens: List[str] = []
    for idx, layer in enumerate(stack.layers):
        alpha, _, _ = layer.euler_deg
        token = f"l{idx}-{_slug_token(layer.material)}-a{int(round(alpha))}"
        if layer.doping.enabled:
            token += f"-drude-wp{int(round(layer.doping.wp_cm1))}-gp{int(round(layer.doping.gp_cm1))}"
        layer_tokens.append(token)
    return f"{_slug_token(mode)}__{'__'.join(layer_tokens)}"[:220]


def _figure_png_bytes(fig: Any) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    buffer.seek(0)
    png_bytes = buffer.getvalue()
    plt.close(fig)
    return png_bytes


def _cached_bytes(cache_key: tuple, builder: Callable[[], bytes]) -> bytes:
    """Memoize expensive export/image byte payloads across reruns.

    The displayed result is immutable once computed, so the same (result,
    settings) key always yields identical bytes. This avoids re-running
    np.savetxt over the full grid and re-rendering the Matplotlib figure on
    every Streamlit rerun (slider drags, widget edits, etc.). The cache is
    bounded so it cannot grow without limit as new results accumulate.
    """
    cache: Dict[tuple, bytes] = st.session_state.setdefault("_export_byte_cache", {})
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    value = builder()
    cache[cache_key] = value
    max_entries = 8
    if len(cache) > max_entries:
        for stale_key in list(cache.keys())[:-max_entries]:
            cache.pop(stale_key, None)
    return value


def _displayed_result_id(mode: str) -> str | None:
    """Stable identifier for the result currently shown, or None if absent."""
    if mode == "Im(rpp) as f(w, kx)":
        if st.session_state.get("map_result") is None:
            return None
        return str(st.session_state.get("selected_map_history_id"))
    if st.session_state.get("iso_result") is None:
        return None
    return str(st.session_state.get("selected_iso_history_id"))


def _dispersion_export_bytes(im_rpp: np.ndarray, kxv_cm1: np.ndarray, wv_cm1: np.ndarray, workers: int, resolution_name: str, stack: StackSpec | None = None) -> bytes:
    kx_grid, w_grid = np.meshgrid(np.asarray(kxv_cm1, dtype=float), np.asarray(wv_cm1, dtype=float), indexing="xy")
    data = np.column_stack(
        [
            np.asarray(im_rpp, dtype=float).reshape(-1),
            kx_grid.reshape(-1),
            w_grid.reshape(-1),
        ]
    )
    stack = stack if stack is not None else build_stack_from_session()
    return _build_export_txt(
        mode="dispersion",
        stack=stack,
        data=data,
        columns=["rpp", "kx", "w"],
        workers=workers,
        resolution_name=resolution_name,
    )


def _isofrequency_export_bytes(
    im_rpp: np.ndarray,
    kxv_cm1: np.ndarray,
    phiv_rad: np.ndarray,
    w0_cm1: float,
    workers: int,
    resolution_name: str,
    stack: StackSpec | None = None,
) -> bytes:
    kx_grid, phi_grid = np.meshgrid(np.asarray(kxv_cm1, dtype=float), np.asarray(phiv_rad, dtype=float), indexing="xy")
    w_grid = np.full_like(kx_grid, float(w0_cm1), dtype=float)
    data = np.column_stack(
        [
            np.asarray(im_rpp, dtype=float).reshape(-1),
            kx_grid.reshape(-1),
            w_grid.reshape(-1),
            phi_grid.reshape(-1),
        ]
    )
    stack = stack if stack is not None else build_stack_from_session()
    return _build_export_txt(
        mode="isofrequency",
        stack=stack,
        data=data,
        columns=["rpp", "kx", "w", "phi"],
        workers=workers,
        resolution_name=resolution_name,
    )


def _dispersion_plot_png_bytes(
    im_rpp: np.ndarray,
    kxv_cm1: np.ndarray,
    wv_cm1: np.ndarray,
    mode_overlay: dict[str, np.ndarray] | None = None,
) -> bytes:
    vmin, vmax = _active_plot_color_limits()
    fig = plot_heatmap(
        x=np.asarray(kx_cm1_to_ui(kxv_cm1), dtype=float),
        y=np.asarray(wv_cm1, dtype=float),
        z=np.asarray(im_rpp, dtype=float),
        xlabel="kx (10^3 cm⁻¹)",
        ylabel="w (cm⁻¹)",
        title="Im(rpp) map",
        cmap=str(PLOT_COLORMAPS[str(st.session_state.get("plot_colormap", "Magma"))]).lower(),
        vmin=vmin,
        vmax=vmax,
        mode_overlay=mode_overlay,
    )
    return _figure_png_bytes(fig)


def _isofrequency_plot_png_bytes(im_rpp: np.ndarray, kxv_cm1: np.ndarray, phiv_rad: np.ndarray) -> bytes:
    vmin, vmax = _active_plot_color_limits()
    fig = plot_polar_isofrequency(
        phi_rad=np.asarray(phiv_rad, dtype=float),
        kx=np.asarray(kx_cm1_to_ui(kxv_cm1), dtype=float),
        z=np.asarray(im_rpp, dtype=float),
        title="Isofrequency Im(rpp) polar surface",
        cmap=str(PLOT_COLORMAPS[str(st.session_state.get("plot_colormap", "Magma"))]).lower(),
        vmin=vmin,
        vmax=vmax,
    )
    return _figure_png_bytes(fig)


def kx_ui_to_cm1(values: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(values, dtype=float) * KX_UI_SCALE


def kx_cm1_to_ui(values: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(values, dtype=float) / KX_UI_SCALE


def _sample_indices(count: int, limit: int) -> List[int]:
    if count <= limit:
        return list(range(count))
    idx = np.linspace(0, count - 1, num=limit, dtype=int)
    return sorted(set(int(i) for i in idx))


def _downsample_grid_for_preview(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    max_x: int,
    max_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_idx = _sample_indices(x.shape[0], max_x)
    y_idx = _sample_indices(y.shape[0], max_y)
    return x[x_idx], y[y_idx], z[np.ix_(y_idx, x_idx)]


def validate_map_inputs(w_min: float, w_max: float, nw: int, kx_min: float, kx_max: float, nk: int) -> List[str]:
    """Validate dispersion-map sampling ranges before enabling computation."""
    errors: List[str] = []
    if w_max <= w_min:
        errors.append("w max must be greater than w min.")
    if kx_max <= kx_min:
        errors.append("kx max must be greater than kx min.")
    if nw <= 0:
        errors.append("Nw must be a positive integer.")
    if nk <= 0:
        errors.append("Nk must be a positive integer.")
    grid_points = int(nw) * int(nk)
    if grid_points > MAX_MAP_GRID_POINTS:
        errors.append(f"Map grid is too dense ({grid_points:,} points). Keep it below {MAX_MAP_GRID_POINTS:,}.")
    return errors


def validate_iso_inputs(
    w0: float,
    kx_min: float,
    kx_max: float,
    nk: int,
    nphi: int,
    phi_min_deg: float,
    phi_max_deg: float,
) -> List[str]:
    """Validate isofrequency sampling ranges before enabling computation."""
    errors: List[str] = []
    if w0 <= 0:
        errors.append("w0 must be greater than 0.")
    if kx_max <= kx_min:
        errors.append("kx max must be greater than kx min.")
    if nk <= 0:
        errors.append("Nk must be a positive integer.")
    if nphi <= 0:
        errors.append("Nphi must be a positive integer.")
    if phi_max_deg <= phi_min_deg:
        errors.append("phi max must be greater than phi min.")
    grid_points = int(nk) * int(nphi)
    if grid_points > MAX_ISO_GRID_POINTS:
        errors.append(f"Isofrequency grid is too dense ({grid_points:,} points). Keep it below {MAX_ISO_GRID_POINTS:,}.")
    return errors


def _render_validation_errors(errors: List[str]) -> None:
    for message in errors:
        st.caption(f":red[• {message}]")


def _finite_color_limit_range(values: np.ndarray) -> tuple[float, float] | None:
    finite_values = np.asarray(values, dtype=float)[np.isfinite(values)]
    if finite_values.size == 0:
        return None
    return float(np.min(finite_values)), float(np.max(finite_values))


def _current_plot_color_limit_source() -> np.ndarray | None:
    mode = str(st.session_state.get("calc_mode", "Im(rpp) as f(w, kx)"))
    if mode == "Im(rpp) as f(w, kx)":
        result = st.session_state.get("map_result")
        if result is None:
            return None
        return np.asarray(result[2], dtype=float)
    result = st.session_state.get("iso_result")
    if result is None:
        return None
    return np.asarray(result[2], dtype=float)


def _seed_manual_color_limits(values: np.ndarray | None = None) -> None:
    source = _current_plot_color_limit_source() if values is None else np.asarray(values, dtype=float)
    if source is None:
        return
    finite_range = _finite_color_limit_range(source)
    if finite_range is None:
        return
    st.session_state.plot_color_min, st.session_state.plot_color_max = finite_range


def _sync_manual_color_limits_toggle() -> None:
    if bool(st.session_state.get("plot_manual_color_limits", False)):
        _seed_manual_color_limits()


def _active_plot_color_limits(mode: str | None = None) -> tuple[float | None, float | None]:
    """Manual limits; with ``mode`` given they only apply to the plot of the target calc mode."""
    if mode is not None and mode != str(st.session_state.get("calc_mode")):
        return None, None
    if not bool(st.session_state.get("plot_manual_color_limits", False)):
        return None, None
    try:
        vmin = float(st.session_state.get("plot_color_min"))
        vmax = float(st.session_state.get("plot_color_max"))
    except (TypeError, ValueError):
        return None, None
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return None, None
    return vmin, vmax


def _manual_color_limits_invalid() -> bool:
    if not bool(st.session_state.get("plot_manual_color_limits", False)):
        return False
    vmin, vmax = _active_plot_color_limits()
    return vmin is None or vmax is None


def _mode_state(mode: str) -> Dict[str, Any]:
    return st.session_state["map_state" if mode == "map" else "iso_state"]


def _sync_mode_state_field(mode: str, field: str, widget_key: str) -> None:
    _mode_state(mode)[field] = st.session_state[widget_key]


def _mode_number_input(mode: str, field: str, label: str, **kwargs: Any) -> None:
    widget_key = f"{mode}_{field}_widget"
    st.session_state[widget_key] = _mode_state(mode)[field]
    st.number_input(
        label,
        key=widget_key,
        on_change=_sync_mode_state_field,
        args=(mode, field, widget_key),
        **kwargs,
    )


def _resolution_choices(speed_presets: Mapping[str, Mapping[str, Mapping[str, int]]]) -> List[str]:
    return [*speed_presets.keys(), "Custom"]


def _apply_map_resolution(resolution_name: str, speed_presets: Mapping[str, Mapping[str, Mapping[str, int]]]) -> None:
    preset = speed_presets[resolution_name]["map"]
    state = _mode_state("map")
    state["nw"] = int(preset["nw"])
    state["nk"] = int(preset["nk"])


def _apply_iso_resolution(resolution_name: str, speed_presets: Mapping[str, Mapping[str, Mapping[str, int]]]) -> None:
    preset = speed_presets[resolution_name]["iso"]
    state = _mode_state("iso")
    state["nk"] = int(preset["nk"])
    state["nphi"] = int(preset["nphi"])


def _sync_resolution_defaults(speed_presets: Mapping[str, Mapping[str, Mapping[str, int]]]) -> None:
    map_resolution = str(st.session_state.get("map_resolution_choice", "Normal"))
    if map_resolution in speed_presets:
        _apply_map_resolution(map_resolution, speed_presets)

    iso_resolution = str(st.session_state.get("iso_resolution_choice", "Normal"))
    if iso_resolution in speed_presets:
        _apply_iso_resolution(iso_resolution, speed_presets)


def _active_resolution_choice(mode: str | None = None) -> str:
    active_mode = mode or str(st.session_state.get("calc_mode", "Im(rpp) as f(w, kx)"))
    if active_mode == "Isofrequency surface":
        return str(st.session_state.get("iso_resolution_choice", "Normal"))
    return str(st.session_state.get("map_resolution_choice", "Normal"))


def _current_calc_settings() -> Dict[str, Any]:
    mode = str(st.session_state.get("calc_mode", "Im(rpp) as f(w, kx)"))
    settings = {
        "mode": mode,
        "resolution": _active_resolution_choice(mode),
        "workers": int(st.session_state.get("worker_count", 1)),
        "fast_preview_plots": bool(st.session_state.get("fast_preview_plots", True)),
    }
    if mode == "Im(rpp) as f(w, kx)":
        settings["params"] = dict(_mode_state("map"))
    else:
        settings["params"] = dict(_mode_state("iso"))
    return settings


def _current_input_signature() -> Dict[str, Any]:
    stack = build_stack_from_session()
    layers = [
        {
            "material": layer.material,
            "thickness_m": float(layer.thickness_m),
            "euler_deg": tuple(float(v) for v in layer.euler_deg),
            "doping": {
                "enabled": bool(layer.doping.enabled),
                "wp_cm1": float(layer.doping.wp_cm1),
                "gp_cm1": float(layer.doping.gp_cm1),
            },
        }
        for layer in stack.layers
    ]
    return {"stack": layers, "calc": _current_calc_settings()}


def _results_are_current() -> bool:
    if st.session_state.get("last_compute_signature") is None:
        return False
    return bool(st.session_state.get("last_compute_signature") == _current_input_signature())


def workspace_status() -> Dict[str, str]:
    """Summarize the current run context for the compact header bar."""
    compute_state = str(st.session_state.get("compute_state", "Idle"))
    freshness = "Current" if _results_are_current() else "Stale"
    mode = "Im(rpp)" if st.session_state.get("calc_mode") == "Im(rpp) as f(w, kx)" else "Isofrequency"
    return {
        "resolution": _active_resolution_choice(),
        "workers": str(st.session_state.get("worker_count", 1)),
        "mode": mode,
        "compute_state": compute_state,
        "freshness": freshness,
    }


def _mode_signature(mode: str) -> Dict[str, Any]:
    """Everything the result of one calculation mode depends on."""
    key = "map" if mode == "Im(rpp) as f(w, kx)" else "iso"
    return {
        "stack": _current_input_signature()["stack"],
        "params": dict(_mode_state(key)),
        "resolution": str(st.session_state.get(f"{key}_resolution_choice", "Normal")),
        "fast": bool(st.session_state.get("fast_solver", True)),
    }


def _store_compute_snapshot(mode: str) -> None:
    st.session_state.setdefault("mode_signatures", {})[mode] = _mode_signature(mode)
    st.session_state.last_compute_signature = _current_input_signature()
    st.session_state.last_compute_mode = mode
    st.session_state.last_compute_timestamp = datetime.now(timezone.utc).isoformat()
    st.session_state.compute_state = "Computed"
    st.session_state.calc_settings = {
        "map": {**dict(_mode_state("map")), "resolution": str(st.session_state.get("map_resolution_choice", "Normal"))},
        "iso": {**dict(_mode_state("iso")), "resolution": str(st.session_state.get("iso_resolution_choice", "Normal"))},
    }


def _stack_history_summary(mode: str) -> str:
    stack = build_stack_from_session()
    materials = [str(layer.material) for layer in stack.layers]
    if not materials:
        stack_summary = "empty stack"
    elif len(materials) <= 4:
        stack_summary = " > ".join(materials)
    else:
        stack_summary = " > ".join(materials[:3]) + f" > ... (+{len(materials) - 3})"

    if mode == "map":
        state = _mode_state("map")
        return (
            f"{stack_summary} | "
            f"phi0 {float(state['phi0']):.1f} deg | "
            f"w {float(state['w_min']):.1f}-{float(state['w_max']):.1f} cm^-1 | "
            f"k {float(state['kx_min']):.3f}-{float(state['kx_max']):.3f} x10^3 cm^-1"
        )

    state = _mode_state("iso")
    return (
        f"{stack_summary} | "
        f"w0 {float(state['w0']):.1f} cm^-1 | "
        f"phi {float(state['phi_min_deg']):.1f}-{float(state['phi_max_deg']):.1f} deg | "
        f"k {float(state['kx_min']):.3f}-{float(state['kx_max']):.3f} x10^3 cm^-1"
    )


def _next_history_id(prefix: str) -> str:
    seq = int(st.session_state.get("history_seq", 0)) + 1
    st.session_state.history_seq = seq
    return f"{prefix}_{seq}"


def _append_map_history(wv: np.ndarray, kxv: np.ndarray, rpp: np.ndarray) -> None:
    w0 = 0.5 * (float(wv[0]) + float(wv[-1]))
    epsilon_superstrate, wave_error = None, None
    try:
        epsilon_superstrate = surface_wave_permittivity(
            build_stack_from_session(), w0, custom_materials=custom_material_registry(),
        )
    except ValueError as exc:
        wave_error = str(exc)
    entry = {
        "rpp": np.asarray(rpp, dtype=complex),
        "phi0_deg": float(_mode_state("map").get("phi0", 0.0)),
        "epsilon_superstrate": epsilon_superstrate,
        "wave_error": wave_error,
        "id": _next_history_id("map"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "resolution": str(st.session_state.get("map_resolution_choice", "Normal")),
        "stack_summary": _stack_history_summary("map"),
        "payload": (np.asarray(wv, dtype=float), np.asarray(kxv, dtype=float), np.imag(rpp).astype(float)),
    }
    st.session_state.map_history.append(entry)
    st.session_state.selected_map_history_id = str(entry["id"])


def _append_iso_history(phiv: np.ndarray, kxv: np.ndarray, rpp: np.ndarray) -> None:
    w0 = float(_mode_state("iso")["w0"])
    epsilon_superstrate = None
    wave_error = None
    try:
        epsilon_superstrate = surface_wave_permittivity(
            build_stack_from_session(), w0, custom_materials=custom_material_registry(),
        )
    except ValueError as exc:
        wave_error = str(exc)
    poles = None
    try:  # cheap (~0.1 s); a failure only disables the pole view
        poles = compute_isofreq_poles(
            build_stack_from_session(), w0, phiv, kxv, rpp, custom_materials=custom_material_registry(),
        )
    except Exception:
        pass
    entry = {
        "id": _next_history_id("iso"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "resolution": str(st.session_state.get("iso_resolution_choice", "Normal")),
        "stack_summary": _stack_history_summary("iso"),
        "payload": (np.asarray(phiv, dtype=float), np.asarray(kxv, dtype=float), np.imag(rpp)),
        "rpp": np.asarray(rpp, dtype=complex),
        "w0_cm1": w0,
        "epsilon_superstrate": epsilon_superstrate,
        "wave_error": wave_error,
        "poles": poles,
    }
    st.session_state.iso_history.append(entry)
    st.session_state.selected_iso_history_id = str(entry["id"])


def _history_label(entry: Mapping[str, Any]) -> str:
    timestamp = str(entry.get("timestamp_utc", ""))
    ts_short = timestamp.replace("T", " ").replace("+00:00", "Z")[:19]
    resolution = str(entry.get("resolution", ""))
    stack_summary = str(entry.get("stack_summary", ""))
    return f"{ts_short} | {resolution} | {stack_summary}"


def _sync_selected_history_result(mode: str) -> None:
    if mode == "map":
        history = list(st.session_state.get("map_history", []))
        if not history:
            return
        selected_id = st.session_state.get("selected_map_history_id")
        if selected_id is None or not any(str(item.get("id")) == str(selected_id) for item in history):
            selected_id = str(history[-1]["id"])
            st.session_state.selected_map_history_id = selected_id
        selected_entry = next((item for item in history if str(item.get("id")) == str(selected_id)), history[-1])
        st.session_state.map_result = tuple(selected_entry["payload"])  # type: ignore[assignment]
        return

    history = list(st.session_state.get("iso_history", []))
    if not history:
        return
    selected_id = st.session_state.get("selected_iso_history_id")
    if selected_id is None or not any(str(item.get("id")) == str(selected_id) for item in history):
        selected_id = str(history[-1]["id"])
        st.session_state.selected_iso_history_id = selected_id
    selected_entry = next((item for item in history if str(item.get("id")) == str(selected_id)), history[-1])
    st.session_state.iso_result = tuple(selected_entry["payload"])  # type: ignore[assignment]


def _render_history_selector(mode: str) -> None:
    if mode == "map":
        history = list(st.session_state.get("map_history", []))
        if not history:
            return
        options = [str(item["id"]) for item in history]
        selected = st.session_state.get("selected_map_history_id")
        if selected not in options:
            selected = options[-1]
        st.session_state.selected_map_history_id = st.selectbox(
            "Saved map",
            options,
            index=options.index(str(selected)),
            format_func=lambda opt: _history_label(next(item for item in history if str(item["id"]) == str(opt))),
        )
        _sync_selected_history_result("map")
        return

    history = list(st.session_state.get("iso_history", []))
    if not history:
        return
    options = [str(item["id"]) for item in history]
    selected = st.session_state.get("selected_iso_history_id")
    if selected not in options:
        selected = options[-1]
    st.session_state.selected_iso_history_id = st.selectbox(
        "Saved isofrequency",
        options,
        index=options.index(str(selected)),
        format_func=lambda opt: _history_label(next(item for item in history if str(item["id"]) == str(opt))),
    )
    _sync_selected_history_result("iso")


def _estimated_cost_label(grid_points: int, workers: int) -> str:
    points_per_worker = max(grid_points / max(workers, 1), 1.0)
    if points_per_worker < 12_000:
        return "Light"
    if points_per_worker < 40_000:
        return "Moderate"
    return "Heavy"


def _run_with_progress(
    start_message: str,
    complete_message: str,
    compute_fn: Callable[[Callable[[float, str], None]], ComputeResultT],
) -> ComputeResultT:
    progress_col, _ = st.columns([0.72, 0.28], gap=None)
    with progress_col:
        progress = st.progress(0.0)
        status = st.empty()

    def _update(fraction: float, message: str) -> None:
        progress.progress(float(np.clip(fraction, 0.0, 1.0)))
        status.caption(f":material/settings: {message}")

    try:
        status.caption(f":material/settings: {start_message}")
        result = compute_fn(_update)
    except Exception:
        st.session_state.compute_state = "Idle"
        progress.empty()
        status.empty()
        raise

    progress.progress(1.0)
    status.caption(f":material/check_circle: {complete_message}")
    return result


def _map_mode_trace_data(
    wv_cm1: np.ndarray, kxv_cm1: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute (and cache) the complex rpp-zero trace for the displayed map.

    The trace is recomputed from the current session stack (with the map's phi0
    pre-tilt) sampled on the displayed frequency/kx grid. Results are cached by the
    displayed result id and stack hash so toggling the overlay does not recompute.
    """
    result_id = _displayed_result_id("Im(rpp) as f(w, kx)")
    if result_id is None:
        return None
    wv = np.asarray(wv_cm1, dtype=float)
    kxv = np.asarray(kxv_cm1, dtype=float)
    if wv.size < 1 or kxv.size < 2:
        return None

    stack = build_stack_from_session()
    phi0 = float(_mode_state("map").get("phi0", 0.0))
    fast = bool(st.session_state.get("fast_solver", True))
    workers = int(st.session_state.get("worker_count", 1))

    cache: Dict[tuple, tuple] = st.session_state.setdefault("_mode_trace_cache", {})
    key = (str(result_id), hash(stack), phi0, fast, int(wv.size), int(kxv.size))
    cached = cache.get(key)
    if cached is None:
        stack_for_run = stack.with_interior_alpha_offset(phi0)
        with st.spinner("Locating optical modes (rpp poles)..."):
            _w, kx_mode, residual, converged = compute_mode_dispersion(
                stack_for_run,
                w_min=float(wv[0]),
                w_max=float(wv[-1]),
                nw=int(wv.size),
                kx_min=float(kxv[0]),
                kx_max=float(kxv[-1]),
                nk=int(kxv.size),
                workers=workers,
                custom_materials=custom_material_registry(),
                fast=fast,
            )
        kx_mode = np.asarray(kx_mode, dtype=np.complex128)
        in_range = (kx_mode.real >= float(kxv[0])) & (kx_mode.real <= float(kxv[-1]))  # drop poles off the map
        cached = (kx_mode, np.asarray(residual, dtype=float), np.asarray(converged, dtype=bool) & in_range)
        cache[key] = cached
        max_entries = 8
        if len(cache) > max_entries:
            for stale_key in list(cache.keys())[:-max_entries]:
                cache.pop(stale_key, None)
    return cached


def _map_mode_trace_overlay(wv_cm1: np.ndarray, kxv_cm1: np.ndarray) -> dict[str, np.ndarray] | None:
    """Build the heatmap overlay dict for the rpp-pole mode trace (UI kx units)."""
    if not _trace_enabled():
        return None
    data = _map_mode_trace_data(wv_cm1, kxv_cm1)
    if data is None:
        return None
    kx_mode, residual, converged = data
    wv = np.asarray(wv_cm1, dtype=float)
    re_ui = np.asarray(kx_cm1_to_ui(np.real(kx_mode)), dtype=float)
    im_ui = np.asarray(kx_cm1_to_ui(np.imag(kx_mode)), dtype=float)
    valid = np.asarray(converged, dtype=bool) & np.isfinite(re_ui)
    if not np.any(valid):
        return None
    # Break the connecting line at frequencies with no genuine mode.
    x = np.where(valid, re_ui, np.nan)
    return {"x": x, "y": wv, "im": im_ui, "residual": np.asarray(residual, dtype=float)}


def _mode_trace_export_bytes(wv_cm1: np.ndarray, kxv_cm1: np.ndarray) -> bytes | None:
    """Serialize the rpp-pole mode trace (w, complex kx, residual) to CSV bytes."""
    data = _map_mode_trace_data(wv_cm1, kxv_cm1)
    if data is None:
        return None
    kx_mode, residual, converged = data
    wv = np.asarray(wv_cm1, dtype=float)
    table = np.column_stack(
        [
            wv,
            np.real(kx_mode),
            np.imag(kx_mode),
            np.asarray(residual, dtype=float),
            np.asarray(converged, dtype=bool).astype(float),
        ]
    )
    buffer = StringIO()
    buffer.write("# mode trace: complex kx of the rpp pole (optical mode), as f(w)\n")
    buffer.write("# units: w=cm^-1, kx=cm^-1 (Re=mode wavevector, Im=loss)\n")
    buffer.write("w,re_kx,im_kx,abs_inv_rpp,converged\n")
    np.savetxt(buffer, table, delimiter=",", fmt="%.12g")
    return buffer.getvalue().encode("utf-8")


def _render_shared_input_strip(speed_presets: Mapping[str, Mapping[str, Mapping[str, int]]]) -> None:
    """Resolution and kx range shared by the dispersion map and the isofrequency surface."""
    map_state, iso_state = _mode_state("map"), _mode_state("iso")
    st.session_state["iso_resolution_choice"] = str(st.session_state.get("map_resolution_choice", "Normal"))
    custom = st.session_state["iso_resolution_choice"] == "Custom"
    st.selectbox("Resolution", _resolution_choices(speed_presets), key="map_resolution_choice")
    if custom:
        cols = st.columns(3, gap="small")
        with cols[0]:
            _mode_number_input("map", "nw", "Nw", min_value=8, step=8)
        with cols[1]:
            _mode_number_input("map", "nk", "Nk", min_value=8, step=8)
        with cols[2]:
            _mode_number_input("iso", "nphi", "Nφ", min_value=8, step=8)
    cols = st.columns(2, gap="small")
    with cols[0]:
        _mode_number_input("map", "kx_min", "kx min (10^3 cm⁻¹)", step=0.05, format="%.3f")
    with cols[1]:
        _mode_number_input("map", "kx_max", "kx max (10^3 cm⁻¹)", step=0.1, format="%.3f")
    iso_state["kx_min"], iso_state["kx_max"] = map_state["kx_min"], map_state["kx_max"]
    iso_state["phi_min_deg"], iso_state["phi_max_deg"] = 0.0, 360.0  # isofrequency always sweeps the full circle
    if custom:
        iso_state["nk"] = map_state["nk"]


def _render_cut_inputs(show_map: bool, show_iso: bool) -> None:
    """φ0 (dispersion cut azimuth) and w0 (isofrequency cut frequency) on one line, then the w range."""
    cols = st.columns(int(show_map) + int(show_iso), gap="small")
    slots = iter(cols)
    if show_map:
        with next(slots):
            _mode_number_input("map", "phi0", "φ0 (deg)", step=1.0, format="%.1f")
    if show_iso:
        with next(slots):
            _mode_number_input("iso", "w0", "w0 (cm⁻¹)", step=5.0)
    if show_map:
        cols = st.columns(2, gap="small")
        with cols[0]:
            _mode_number_input("map", "w_min", "w min (cm⁻¹)", step=10.0)
        with cols[1]:
            _mode_number_input("map", "w_max", "w max (cm⁻¹)", step=10.0)


def _active_compute_summary() -> tuple[List[str], List[str], int, str]:
    e1, w1, n1, _ = _compute_summary("Im(rpp) as f(w, kx)")
    e2, w2, n2, _ = _compute_summary("Isofrequency surface")
    return e1 + e2, w1 + w2, n1 + n2, ":material/play_arrow: Compute both"


def _compute_summary(mode: str) -> tuple[List[str], List[str], int, str]:
    stack_materials = [layer.material for layer in build_stack_from_session().layers]
    if mode == "Im(rpp) as f(w, kx)":
        map_state = _mode_state("map")
        errors = validate_map_inputs(
            w_min=float(map_state["w_min"]),
            w_max=float(map_state["w_max"]),
            nw=int(map_state["nw"]),
            kx_min=float(map_state["kx_min"]),
            kx_max=float(map_state["kx_max"]),
            nk=int(map_state["nk"]),
        )
        warnings = material_frequency_range_warnings(
            stack_materials,
            custom_material_registry(),
            w_min_cm1=float(map_state["w_min"]),
            w_max_cm1=float(map_state["w_max"]),
        )
        return errors, warnings, int(map_state["nw"]) * int(map_state["nk"]), ":material/play_arrow: Compute Im(rpp) map"

    iso_state = _mode_state("iso")
    errors = validate_iso_inputs(
        w0=float(iso_state["w0"]),
        kx_min=float(iso_state["kx_min"]),
        kx_max=float(iso_state["kx_max"]),
        nk=int(iso_state["nk"]),
        nphi=int(iso_state["nphi"]),
        phi_min_deg=float(iso_state["phi_min_deg"]),
        phi_max_deg=float(iso_state["phi_max_deg"]),
    )
    warnings = material_frequency_range_warnings(
        stack_materials,
        custom_material_registry(),
        w_min_cm1=float(iso_state["w0"]),
        w_max_cm1=float(iso_state["w0"]),
    )
    return errors, warnings, int(iso_state["nk"]) * int(iso_state["nphi"]), ":material/play_arrow: Compute isofrequency Im(rpp)"


def _execute_active_compute(workers: int) -> None:
    modes = ("Im(rpp) as f(w, kx)", "Isofrequency surface")
    # Recompute only what changed (e.g. just phi0 or w0); an unchanged click reruns both.
    stale = [m for m in modes if st.session_state.get("mode_signatures", {}).get(m) != _mode_signature(m)]
    for mode in stale or modes:
        _execute_compute(mode, workers)


def _execute_compute(mode: str, workers: int) -> None:
    fast = bool(st.session_state.get("fast_solver", True))
    st.session_state.compute_state = "Running"
    stack = build_stack_from_session()
    if mode == "Im(rpp) as f(w, kx)":
        map_state = _mode_state("map")
        stack_for_run = stack.with_interior_alpha_offset(float(map_state.get("phi0", 0.0)))
        wv, kxv, rpp_map = _run_with_progress(
            start_message="Preparing dispersion computation...",
            complete_message="Dispersion computation complete",
            compute_fn=lambda progress_cb: compute_rpp_map(
                stack_for_run,
                w_min=float(map_state["w_min"]),
                w_max=float(map_state["w_max"]),
                nw=int(map_state["nw"]),
                kx_min=float(kx_ui_to_cm1(map_state["kx_min"])),
                kx_max=float(kx_ui_to_cm1(map_state["kx_max"])),
                nk=int(map_state["nk"]),
                workers=int(workers),
                progress=progress_cb,
                custom_materials=custom_material_registry(),
                fast=fast,
                return_complex=True,
            ),
        )
        _append_map_history(wv, kxv, rpp_map)
        _sync_selected_history_result("map")
        _store_compute_snapshot("Im(rpp) as f(w, kx)")
        return

    iso_state = _mode_state("iso")
    phiv, kxv, rpp = _run_with_progress(
        start_message="Preparing isofrequency computation...",
        complete_message="Isofrequency computation complete",
        compute_fn=lambda progress_cb: compute_isofreq_map(
            stack,
            w0=float(iso_state["w0"]),
            kx_min=float(kx_ui_to_cm1(iso_state["kx_min"])),
            kx_max=float(kx_ui_to_cm1(iso_state["kx_max"])),
            nk=int(iso_state["nk"]),
            nphi=int(iso_state["nphi"]),
            phi_min_deg=float(iso_state["phi_min_deg"]),
            phi_max_deg=float(iso_state["phi_max_deg"]),
            global_phi_sweep=True,
            workers=int(workers),
            progress=progress_cb,
            custom_materials=custom_material_registry(),
            fast=fast,
            return_complex=True,
        ),
    )
    _append_iso_history(phiv, kxv, rpp)
    _sync_selected_history_result("iso")
    _store_compute_snapshot("Isofrequency surface")


def render_run_controls_panel(
    speed_presets: Mapping[str, Mapping[str, Mapping[str, int]]],
    between: Callable[[], None] | None = None,
) -> None:
    """Render mode selector, optional ``between`` inputs, solver toggle and compute action."""
    _sync_resolution_defaults(speed_presets)
    if between is not None:
        between()
    workers = int(st.session_state.get("worker_count", 1))
    st.session_state.setdefault("fast_solver", True)
    st.toggle(
        "Fast vectorised solver",
        key="fast_solver",
        help=(
            "Evaluate each frequency/angle row with a vectorised transfer-matrix "
            "solver instead of the per-point loop. Results are numerically identical "
            "to the standard solver (matching to floating-point round-off); it is "
            "typically 10-20x faster on dense grids. Any row it cannot handle falls "
            "back automatically to the exact per-point evaluation. Turn off to force "
            "the reference per-point solver."
        ),
    )
    errors, warnings, _, button_label = _active_compute_summary()
    for warning in warnings:
        st.warning(warning, icon=":material/warning:")
    _render_validation_errors(errors)
    with st.form("run_controls_form", border=False):
        run_now = st.form_submit_button(button_label, width="stretch", type="primary", disabled=bool(errors))
    if run_now:
        _execute_active_compute(workers=workers)


def render_mode_input_strip(speed_presets: Mapping[str, Mapping[str, Mapping[str, int]]]) -> None:
    """Render only the active mode-specific sampling inputs."""
    _sync_resolution_defaults(speed_presets)
    _render_shared_input_strip(speed_presets)
    _render_cut_inputs(True, True)


def render_wave_options() -> None:
    """Real-space wave controls (right column)."""
    st.radio("Method", WAVE_METHODS, key="wave_method")
    cols = st.columns(2)
    with cols[0]:
        st.number_input("Source height (nm)", min_value=0.0, step=5.0, key="wave_source_height_nm")
        st.number_input("View half-width (µm)", min_value=0.1, step=0.5, key="wave_half_width_um")
    with cols[1]:
        st.number_input("Observation height (nm)", min_value=0.0, step=5.0, key="wave_observation_height_nm")
        st.selectbox(
            "Fourier grid", [512, 1024, 2048], key="wave_fft_size",
            disabled=str(st.session_state.get("wave_method")) == WAVE_METHODS[1],
        )
    st.radio("Display", WAVE_DISPLAYS, key="wave_display")


def _trace_enabled() -> bool:
    """The pole trace is drawn (and exportable) whenever the real-space method is "Poles"."""
    return str(st.session_state.get("wave_method")) == WAVE_METHODS[1]


def _render_plot_toolbar(
    has_result: bool,
    export_bytes: bytes | None,
    file_name: str | None,
    image_bytes: bytes | None,
    image_file_name: str | None,
) -> None:
    with st.container():
        st.caption(":material/tune: Plot controls")
        settings_row = st.columns([0.22, 0.2, 0.14, 0.14, 0.14, 0.16], gap=None, vertical_alignment="bottom")
        with settings_row[0]:
            st.selectbox("Colormap", list(PLOT_COLORMAPS.keys()), key="plot_colormap")
        with settings_row[1]:
            st.toggle(
                "Manual colorscale",
                key="plot_manual_color_limits",
                disabled=not has_result,
                on_change=_sync_manual_color_limits_toggle,
            )
        with settings_row[2]:
            st.number_input(
                "Min",
                key="plot_color_min",
                disabled=not has_result or not bool(st.session_state.get("plot_manual_color_limits", False)),
            )
        with settings_row[3]:
            st.number_input(
                "Max",
                key="plot_color_max",
                disabled=not has_result or not bool(st.session_state.get("plot_manual_color_limits", False)),
            )
        with settings_row[4]:
            st.toggle(
                "Preview plots",
                key="fast_preview_plots",
                help="Downsamples only the displayed plot. Computation stays full resolution.",
            )
        with settings_row[5]:
            st.selectbox(
                "Export / limits for",
                ["Im(rpp) as f(w, kx)", "Isofrequency surface"],
                format_func=lambda m: "Dispersion" if m.startswith("Im") else "Isofrequency",
                key="calc_mode",
            )
        if _manual_color_limits_invalid():
            st.caption(":red[Manual colorscale requires Max > Min. Plot display falls back to auto until fixed.]")

        is_map_mode = str(st.session_state.get("calc_mode", "Im(rpp) as f(w, kx)")) == "Im(rpp) as f(w, kx)"
        map_has_result = is_map_mode and st.session_state.get("map_result") is not None
        if map_has_result and _trace_enabled():
            wv, kxv, _im = st.session_state.map_result
            trace_bytes = _mode_trace_export_bytes(wv, kxv)
            if trace_bytes is not None:
                st.download_button(
                    ":material/download: Export mode trace",
                    data=trace_bytes,
                    file_name=f"{_stack_plot_filename_stem('mode-trace-rpp-pole')}.csv",
                    mime="text/csv",
                    width="stretch",
                )
            else:
                st.caption(":material/info: No optical mode found on the current grid.")

        action_row = st.columns([0.22, 0.26, 0.26, 0.26], gap=None, vertical_alignment="center")
        with action_row[0]:
            st.caption(":material/settings: Actions")
        with action_row[1]:
            if st.button(":material/refresh: Reset view", width="stretch", disabled=not has_result):
                st.session_state.plot_refresh_nonce = int(st.session_state.get("plot_refresh_nonce", 0)) + 1
        with action_row[2]:
            if has_result and image_bytes is not None and image_file_name is not None:
                st.download_button(
                    ":material/image: Export image",
                    data=image_bytes,
                    file_name=image_file_name,
                    mime="image/png",
                    width="stretch",
                )
            else:
                st.button(":material/image: Export image", width="stretch", disabled=True)
        with action_row[3]:
            if has_result and export_bytes is not None and file_name is not None:
                st.download_button(
                    ":material/download: Export data",
                    data=export_bytes,
                    file_name=file_name,
                    mime="text/plain",
                    width="stretch",
                    type="primary",
                )
            else:
                st.button(":material/download: Export data", width="stretch", disabled=True)


def _current_export_payload(resolution_name: str, workers: int) -> tuple[bytes | None, str | None]:
    mode = str(st.session_state.get("calc_mode", "Im(rpp) as f(w, kx)"))
    result_id = _displayed_result_id(mode)
    if mode == "Im(rpp) as f(w, kx)" and st.session_state.map_result is not None:
        wv, kxv, im_rpp = st.session_state.map_result
        stack = build_stack_from_session()
        key = ("csv-map", result_id, workers, resolution_name, hash(stack))
        return (
            _cached_bytes(
                key,
                lambda: _dispersion_export_bytes(
                    im_rpp=im_rpp, kxv_cm1=kxv, wv_cm1=wv, workers=workers, resolution_name=resolution_name, stack=stack
                ),
            ),
            f"{_stack_plot_filename_stem('dispersion-rpp-export', stack)}.txt",
        )
    if mode == "Isofrequency surface" and st.session_state.iso_result is not None:
        phiv, kxv, im_rpp = st.session_state.iso_result
        stack = build_stack_from_session()
        w0 = float(_mode_state("iso")["w0"])
        key = ("csv-iso", result_id, workers, resolution_name, w0, hash(stack))
        return (
            _cached_bytes(
                key,
                lambda: _isofrequency_export_bytes(
                    im_rpp=im_rpp,
                    kxv_cm1=kxv,
                    phiv_rad=phiv,
                    w0_cm1=w0,
                    workers=workers,
                    resolution_name=resolution_name,
                    stack=stack,
                ),
            ),
            f"{_stack_plot_filename_stem('isofrequency-rpp-export', stack)}.txt",
        )
    return None, None


def _current_plot_image_payload() -> tuple[bytes | None, str | None]:
    mode = str(st.session_state.get("calc_mode", "Im(rpp) as f(w, kx)"))
    result_id = _displayed_result_id(mode)
    colormap = str(st.session_state.get("plot_colormap", "Magma"))
    vmin, vmax = _active_plot_color_limits()
    if mode == "Im(rpp) as f(w, kx)" and st.session_state.map_result is not None:
        wv, kxv, im_rpp = st.session_state.map_result
        mode_overlay = _map_mode_trace_overlay(wv, kxv)
        overlay_key = None if mode_overlay is None else hash(np.asarray(mode_overlay["x"], dtype=float).tobytes())
        key = ("png-map", result_id, colormap, vmin, vmax, overlay_key)
        return (
            _cached_bytes(
                key,
                lambda: _dispersion_plot_png_bytes(im_rpp=im_rpp, kxv_cm1=kxv, wv_cm1=wv, mode_overlay=mode_overlay),
            ),
            f"{_stack_plot_filename_stem('im-rpp-map')}.png",
        )
    if mode == "Isofrequency surface" and st.session_state.iso_result is not None:
        phiv, kxv, im_rpp = st.session_state.iso_result
        key = ("png-iso", result_id, colormap, vmin, vmax)
        return (
            _cached_bytes(key, lambda: _isofrequency_plot_png_bytes(im_rpp=im_rpp, kxv_cm1=kxv, phiv_rad=phiv)),
            f"{_stack_plot_filename_stem('isofrequency-im-rpp')}.png",
        )
    return None, None


def _render_map_plot(resolution_name: str, workers: int) -> None:
    if st.session_state.map_result is None:
        st.caption(":material/analytics: Run the computation to show the map.")
        return None
    wv, kxv, im_rpp = st.session_state.map_result
    full_kx_ui = np.asarray(kx_cm1_to_ui(kxv), dtype=float)
    full_w = np.asarray(wv, dtype=float)
    full_z = np.asarray(im_rpp, dtype=float)
    mode_overlay = _map_mode_trace_overlay(wv, kxv)
    fast_preview = bool(st.session_state.get("fast_preview_plots", True))
    preview_caption: str | None = None
    if fast_preview:
        kx_plot, w_plot, z_plot = _downsample_grid_for_preview(
            full_kx_ui,
            full_w,
            full_z,
            max_x=FAST_PREVIEW_MAX_X,
            max_y=FAST_PREVIEW_MAX_Y,
        )
        preview_caption = f"Preview: {z_plot.shape[0]}x{z_plot.shape[1]} shown from {im_rpp.shape[0]}x{im_rpp.shape[1]}."
    else:
        kx_plot = full_kx_ui
        w_plot = full_w
        z_plot = full_z
    vmin, vmax = _active_plot_color_limits("Im(rpp) as f(w, kx)")
    fig = plot_heatmap_interactive(
        kx_plot,
        w_plot,
        z_plot,
        xlabel="kx (10^3 cm⁻¹)",
        ylabel="w (cm⁻¹)",
        title="Im(rpp) map",
        cmap=PLOT_COLORMAPS[str(st.session_state.get("plot_colormap", "Magma"))],
        height=_plot_height(455),
        zmin=vmin,
        zmax=vmax,
        mode_overlay=mode_overlay,
        hline=_marker_w0(),
        hline_label="ω₀ (isofrequency)",
    )
    st.plotly_chart(
        fig,
        width="stretch",
        config=PLOTLY_CONFIG,
        key=f"map_plot_{st.session_state.get('plot_refresh_nonce', 0)}",
    )
    if preview_caption:
        _note(preview_caption)


def _render_iso_plot(resolution_name: str, workers: int) -> None:
    if st.session_state.iso_result is None:
        st.caption(":material/radar: Run the computation to show the surface.")
        return None
    phiv, kxv, im_rpp = st.session_state.iso_result
    full_phi = np.asarray(phiv, dtype=float)
    full_kx_ui = np.asarray(kx_cm1_to_ui(kxv), dtype=float)
    full_z = np.asarray(im_rpp, dtype=float)
    fast_preview = bool(st.session_state.get("fast_preview_plots", True))
    preview_caption: str | None = None
    if fast_preview:
        kx_plot, phi_plot, z_plot = _downsample_grid_for_preview(
            full_kx_ui,
            full_phi,
            full_z,
            max_x=FAST_PREVIEW_MAX_R,
            max_y=FAST_PREVIEW_MAX_PHI,
        )
        preview_caption = f"Preview: {z_plot.shape[0]}x{z_plot.shape[1]} shown from {im_rpp.shape[0]}x{im_rpp.shape[1]}."
    else:
        phi_plot = full_phi
        kx_plot = full_kx_ui
        z_plot = full_z
    vmin, vmax = _active_plot_color_limits("Isofrequency surface")
    fig = plot_polar_isofrequency_interactive(
        phi_plot,
        kx_plot,
        z_plot,
        title="Isofrequency Im(rpp) polar surface",
        radial_label="kx (10^3 cm^-1)",
        cmap=PLOT_COLORMAPS[str(st.session_state.get("plot_colormap", "Magma"))],
        height=_plot_height(455),
        cmin=vmin,
        cmax=vmax,
        phi_line_deg=_marker_phi0_deg(),
    )
    st.plotly_chart(
        fig,
        width="stretch",
        config=PLOTLY_CONFIG,
        key=f"iso_plot_{st.session_state.get('plot_refresh_nonce', 0)}",
    )
    if preview_caption:
        _note(preview_caption)


def _note(text: str) -> None:
    """Explanatory caption under a plot; omitted in the compact 2×2 grid."""
    return  # the 2×2 grid shows titles only


def _plot_height(single: int) -> int:
    return 330


def _marker_w0() -> float | None:
    """Frequency of the isofrequency cut (latest saved run, else the current input)."""
    entry = _selected_history_entry("iso")
    return float(entry["w0_cm1"]) if entry is not None and "w0_cm1" in entry else float(_mode_state("iso").get("w0", 0.0))


def _marker_phi0_deg() -> float:
    """Azimuth of the dispersion cut (latest saved map, else the current input)."""
    entry = _selected_history_entry("map")
    return float(entry["phi0_deg"]) if entry is not None and "phi0_deg" in entry else float(_mode_state("map").get("phi0", 0.0))


def _selected_history_entry(mode: str) -> Mapping[str, Any] | None:
    key, hist = ("selected_map_history_id", "map_history") if mode == "map" else ("selected_iso_history_id", "iso_history")
    selected_id = st.session_state.get(key)
    return next((item for item in st.session_state.get(hist, []) if item["id"] == selected_id), None)


def _show_wave_heatmap(
    x: np.ndarray, y: np.ndarray, field: np.ndarray, intensity: np.ndarray, *,
    xlabel: str, ylabel: str, square: bool, height: int, key: str,
    hline: float | None = None, ray_deg: float | None = None,
) -> None:
    display = str(st.session_state.get("wave_display", WAVE_DISPLAYS[0]))
    cmap = PLOT_COLORMAPS[str(st.session_state.get("plot_colormap", "Magma"))]
    if display == WAVE_DISPLAYS[0]:
        shown = field.real
        # The dipole peak is orders of magnitude above the wavefronts: clip at the 98th percentile.
        lim = float(np.percentile(np.abs(shown), 98)) or 1.0
        zmin, zmax, cmap, colorbar, title = -lim, lim, "RdBu", "Re(Ez) (norm.)", "Real-space field Re(Ez)"
    elif display == WAVE_DISPLAYS[1]:
        shown = 10 * np.log10(np.maximum(intensity, 1e-8))
        zmin, zmax, colorbar, title = -60.0, 0.0, "Intensity (dB)", "Real-space intensity |Ez|²"
    else:
        shown = intensity
        zmin, zmax, colorbar, title = 0.0, 1.0, "Normalized |Ez|²", "Real-space intensity |Ez|²"
    fig = plot_heatmap_interactive(
        x, y, shown, xlabel=xlabel, ylabel=ylabel, title=title, cmap=cmap,
        height=_plot_height(height), zmin=zmin, zmax=zmax, colorbar_title=colorbar,
        hline=hline, hline_label="ω₀ (isofrequency)",
    )
    fig.update_traces(selector=dict(type="heatmap"), hovertemplate=f"x=%{{x:.3f}} µm<br>y=%{{y:.4g}}<br>Value=%{{z:.4g}}<extra></extra>")
    if square:
        fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")
        fig.update_xaxes(constrain="domain")
        fig.add_scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="x", size=8, color="#5ec8f2"),
                        name="Dipole", hovertemplate="Dipole at (0, 0)<extra></extra>")
        if ray_deg is not None:
            reach = float(np.max(np.abs(x)))
            fig.add_scatter(
                x=[0, reach * np.cos(np.deg2rad(ray_deg))], y=[0, reach * np.sin(np.deg2rad(ray_deg))],
                mode="lines", line=dict(color=MARKER_LINE_COLOR, width=1.5, dash="dash"),
                name=f"phi0 = {ray_deg:.1f} deg", hoverinfo="name", showlegend=False,
            )
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG,
                    key=f"{key}_{st.session_state.get('plot_refresh_nonce', 0)}")


def _render_map_wave_plot() -> None:
    """Real-space line cut x vs w of the dispersion map: Fourier transform of rpp or one plane wave per pole."""
    entry = _selected_history_entry("map")
    if entry is None or "rpp" not in entry:
        st.info("Recompute the dispersion map to save the complex response for the real-space view.")
        return
    wv, kxv, _ = entry["payload"]
    half_width = float(st.session_state.wave_half_width_um)
    use_poles = str(st.session_state.get("wave_method")) == WAVE_METHODS[1]
    try:
        if use_poles:
            data = _map_mode_trace_data(wv, kxv)
            if data is None:
                st.info("No mode trace available.")
                return
            kx_mode, _res, ok = data
            xs, field = compute_dispersion_wave_poles(kx_mode, ok, half_width_um=half_width)
            intensity = np.abs(field) ** 2
        else:
            if entry.get("wave_error"):
                st.info(entry["wave_error"])
                return
            xs, field = compute_dispersion_wave_ft(
                wv, kxv, entry["rpp"],
                source_height_nm=float(st.session_state.wave_source_height_nm),
                observation_height_nm=float(st.session_state.wave_observation_height_nm),
                epsilon_superstrate=entry["epsilon_superstrate"], half_width_um=half_width,
            )
            intensity = np.abs(field) ** 2
    except ValueError as exc:
        st.info(str(exc))
        return
    _show_wave_heatmap(xs, wv, field, intensity, xlabel="x (µm)", ylabel="w (cm⁻¹)", square=False, height=455,
                       key="wave_map_plot", hline=_marker_w0())
    _note(
        "Line cut y = 0, x > 0. "
        + ("One plane wave exp(i·kx·x) per frequency from the complex pole kx (unit amplitude at x = 0)."
           if use_poles else
           "Fourier transform of the evanescent part of rpp(w, kx) over the map's kx range; each row normalised "
           "outside the source core. kx > 0 samples only, so a finite kx range limits the resolution.")
    )


def _render_surface_wave_plot() -> None:
    """Reconstruct the wave from the selected run, independently of edited inputs."""
    entry = _selected_history_entry("iso")
    if entry is None or "rpp" not in entry:
        st.info("Recompute isofrequency to save the complex response for the wave map.")
        return
    if entry.get("wave_error"):
        st.info(entry["wave_error"])
        return
    poles = entry.get("poles")
    use_poles = poles is not None and str(st.session_state.get("wave_method")) == WAVE_METHODS[1]
    phi, k, _ = entry["payload"]
    try:
        if use_poles:
            xy, field, intensity = compute_pole_wave(
                phi, *poles, w0_cm1=entry["w0_cm1"],
                source_height_nm=float(st.session_state.wave_source_height_nm),
                observation_height_nm=float(st.session_state.wave_observation_height_nm),
                epsilon_superstrate=entry["epsilon_superstrate"],
                half_width_um=float(st.session_state.wave_half_width_um),
            )
        else:
            xy, field, intensity = compute_surface_wave(
                phi, k, entry["rpp"], w0_cm1=entry["w0_cm1"],
                source_height_nm=float(st.session_state.wave_source_height_nm),
                observation_height_nm=float(st.session_state.wave_observation_height_nm),
                epsilon_superstrate=entry["epsilon_superstrate"],
                fft_size=int(st.session_state.wave_fft_size),
            )
    except ValueError as exc:
        st.info(str(exc))
        return
    half_width = float(st.session_state.wave_half_width_um)
    # Show the central half of the periodic FFT domain to keep its edges out of view.
    safe_half_width = abs(xy[0]) if use_poles else abs(xy[0]) / 2
    if half_width > safe_half_width:
        st.info(f"View limited to ±{safe_half_width:.2f} µm. Increase the Fourier grid for a larger view.")
        half_width = safe_half_width
    keep = np.abs(xy) <= half_width
    if np.count_nonzero(keep) < 3:
        st.info("Increase the view half-width or the isofrequency maximum momentum to resolve the view.")
        return
    if not np.any(intensity):
        st.info("No evanescent response in the sampled momentum band; extend the isofrequency range.")
    sub = np.ix_(keep, keep)
    _show_wave_heatmap(xy[keep], xy[keep], field[sub], intensity[sub], xlabel="x (µm)", ylabel="y (µm)",
                       square=True, height=510, key="wave_iso_plot", ray_deg=_marker_phi0_deg())
    if use_poles:
        note = ("Pole (residue) part of the strongest mode branch only: no background or higher-order "
                "branches; azimuths without a converged pole are blank.")
    else:
        note = (f"Normalized evanescent scattered |Ez|²; includes nonresonant background. Fourier period: "
                f"{len(xy) * (xy[1] - xy[0]):.1f} µm. Refine Nk, Nphi and the Fourier grid, and extend the "
                "momentum range to check convergence.")
    _note(f"Saved run: {entry['w0_cm1']:.1f} cm⁻¹. {note} Excludes the direct source field.")


def _render_metadata_tab() -> None:
    settings = _current_calc_settings()
    st.caption("Run context")
    st.json(settings, expanded=False)
    st.caption("Snapshot")
    st.json(
        {
            "compute_state": st.session_state.get("compute_state", "Idle"),
            "last_compute_mode": st.session_state.get("last_compute_mode"),
            "last_compute_timestamp": st.session_state.get("last_compute_timestamp"),
            "freshness": "Current" if _results_are_current() else "Stale",
        },
        expanded=False,
    )


def _render_export_tab(resolution_name: str, workers: int) -> None:
    mode = str(st.session_state.get("calc_mode", "Im(rpp) as f(w, kx)"))
    export_bytes, export_name = _current_export_payload(resolution_name=resolution_name, workers=workers)
    image_bytes, image_name = _current_plot_image_payload()
    if export_bytes is None or image_bytes is None:
        st.caption("No computed result is available for export yet.")
        return
    data_label = "Download dispersion export" if mode == "Im(rpp) as f(w, kx)" else "Download isofrequency export"
    st.download_button(
        data_label,
        data=export_bytes,
        file_name=export_name,
        mime="text/plain",
        width="stretch",
    )
    st.download_button(
        "Download plot image",
        data=image_bytes,
        file_name=image_name,
        mime="image/png",
        width="stretch",
    )


def render_results_panel() -> None:
    """Render the main plot, status tabs, and export actions."""
    resolution_name = _active_resolution_choice()
    workers = int(st.session_state.get("worker_count", 1))
    mode = str(st.session_state.get("calc_mode", "Im(rpp) as f(w, kx)"))
    _sync_selected_history_result("map")
    _sync_selected_history_result("iso")

    with st.container(gap=None):
        cells = [
            [_render_map_plot, lambda *_: _render_map_wave_plot()],
            [_render_iso_plot, lambda *_: _render_surface_wave_plot()],
        ]
        for row in cells:
            for col, render in zip(st.columns(2, gap="small"), row):
                with col, st.container(border=True):
                    render(resolution_name, workers)
        with st.container(border=True):
            for col, kind in zip(st.columns(2, gap="small"), ("map", "iso")):
                with col:
                    _render_history_selector(kind)

        with st.container(border=True):
            has_result = (mode == "Im(rpp) as f(w, kx)" and st.session_state.map_result is not None) or (
                mode == "Isofrequency surface" and st.session_state.iso_result is not None
            )
            export_bytes, export_name = _current_export_payload(resolution_name=resolution_name, workers=workers)
            image_bytes, image_name = _current_plot_image_payload()
            _render_plot_toolbar(
                has_result=has_result,
                export_bytes=export_bytes,
                file_name=export_name,
                image_bytes=image_bytes,
                image_file_name=image_name,
            )

        with st.container(border=True):
            tab_result, tab_meta, tab_export = st.tabs(["Status", "Metadata", "Export"])
            with tab_result:
                st.markdown(
                    f"**Displayed mode:** {'Im(rpp) map' if mode == 'Im(rpp) as f(w, kx)' else 'Isofrequency surface'}  \n"
                    f"**Plot freshness:** {'Current' if _results_are_current() else 'Stale'}"
                )
                if st.session_state.get("last_compute_timestamp"):
                    st.caption(f"Last compute timestamp (UTC): {st.session_state['last_compute_timestamp']}")
            with tab_meta:
                _render_metadata_tab()
            with tab_export:
                _render_export_tab(resolution_name=resolution_name, workers=workers)
