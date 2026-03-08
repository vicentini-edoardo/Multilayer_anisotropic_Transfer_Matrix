"""Calculation setup, result rendering, and export utilities for the Streamlit UI."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from io import StringIO
import re
from typing import Any, Callable, Dict, List, Mapping, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..plotting import (
    plot_heatmap,
    plot_heatmap_interactive,
    plot_polar_isofrequency,
    plot_polar_isofrequency_interactive,
    plot_stack_pseudo3d,
)
from ..solver import compute_isofreq_map, compute_rpp_map
from .layer_builder import build_stack_from_session
from .theme_layout import single_choice
from ..models import StackSpec


MAX_MAP_GRID_POINTS = 250_000
MAX_ISO_GRID_POINTS = 250_000
KX_UI_SCALE = 1e3
FAST_PREVIEW_MAX_X = 220
FAST_PREVIEW_MAX_Y = 220
FAST_PREVIEW_MAX_R = 220
FAST_PREVIEW_MAX_PHI = 220
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
    "speed_preset_choice",
    "worker_count",
    "fast_preview_plots",
    "advanced_euler_enabled",
    "plot_colormap",
    "show_peak_dots",
    "peak_dot_threshold_percent",
    "map_w_min",
    "map_w_max",
    "map_nw",
    "map_kx_min",
    "map_kx_max",
    "map_nk",
    "iso_w0",
    "iso_nk",
    "iso_nphi",
    "iso_kx_min",
    "iso_kx_max",
)

def _initial_calc_defaults(speed_presets: Mapping[str, Mapping[str, int]]) -> Dict[str, Any]:
    default_preset = "Normal" if "Normal" in speed_presets else next(iter(speed_presets))
    return {
        "calc_mode": "Im(rpp) as f(w, kx)",
        "speed_preset_choice": default_preset,
        "worker_count": 4,
        "fast_preview_plots": True,
        "advanced_euler_enabled": False,
        "plot_colormap": "Magma",
        "show_peak_dots": True,
        "peak_dot_threshold_percent": 10.0,
        "plot_refresh_nonce": 0,
        "compute_state": "Idle",
        "last_compute_signature": None,
        "last_compute_mode": None,
        "last_compute_timestamp": None,
        "map_result": None,
        "iso_result": None,
        "calc_settings": {
            "map": {
                "w_min": 780.0,
                "w_max": 1050.0,
                "nw": int(speed_presets[default_preset]["nw"]),
                "kx_min": 0.05,
                "kx_max": 15.0,
                "nk": int(speed_presets[default_preset]["nk"]),
            },
            "iso": {
                "w0": 500.0,
                "nk": int(speed_presets[default_preset]["nk"]),
                "nphi": int(speed_presets[default_preset]["nphi"]),
                "kx_min": 0.05,
                "kx_max": 15.0,
            },
        },
        "map_w_min": 780.0,
        "map_w_max": 1050.0,
        "map_nw": int(speed_presets[default_preset]["nw"]),
        "map_kx_min": 0.05,
        "map_kx_max": 15.0,
        "map_nk": int(speed_presets[default_preset]["nk"]),
        "iso_w0": 500.0,
        "iso_nk": int(speed_presets[default_preset]["nk"]),
        "iso_nphi": int(speed_presets[default_preset]["nphi"]),
        "iso_kx_min": 0.05,
        "iso_kx_max": 15.0,
        "last_applied_speed_preset": default_preset,
    }


def _calc_ui_state_snapshot() -> Dict[str, Any]:
    return {key: st.session_state[key] for key in CALC_UI_STATE_KEYS if key in st.session_state}


def init_calculation_state(speed_presets: Mapping[str, Mapping[str, int]]) -> None:
    """Initialize persistent calculation settings and restore them across reruns."""
    defaults = _initial_calc_defaults(speed_presets)
    preserved_state = st.session_state.pop("_preserved_calc_state", None) or {}
    mirrored_state = st.session_state.get("_calc_ui_state_mirror", {})
    restored_state = {**mirrored_state, **preserved_state}

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = restored_state.get(key, default_value)

    if st.session_state.get("speed_preset_choice") not in speed_presets:
        st.session_state.speed_preset_choice = str(defaults["speed_preset_choice"])
    if st.session_state.get("calc_mode") not in ("Im(rpp) as f(w, kx)", "Isofrequency surface"):
        st.session_state.calc_mode = str(defaults["calc_mode"])

    st.session_state["_calc_ui_state_mirror"] = _calc_ui_state_snapshot()


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
    preset_name: str,
) -> bytes:
    buffer = StringIO()
    now = datetime.now(timezone.utc).isoformat()
    buffer.write(f"# exported_utc: {now}\n")
    buffer.write(f"# mode: {mode}\n")
    buffer.write(f"# preset: {preset_name}\n")
    buffer.write(f"# workers: {workers}\n")
    buffer.write("# units: rpp=Im(rpp), kx=cm^-1, w=cm^-1, phi=rad\n")
    buffer.write(f"# rows: {data.shape[0]}\n")
    buffer.write(f"# columns: {','.join(columns)}\n")
    buffer.write("# layers:\n")
    buffer.write("# layer_index,role,material,thickness_m,thickness_um,alpha_deg,beta_deg,gamma_deg\n")
    total_layers = len(stack.layers)
    for idx, layer in enumerate(stack.layers):
        alpha, beta, gamma = layer.euler_deg
        role = _stack_role(idx, total_layers)
        thickness_um = float(layer.thickness_m) * 1e6
        buffer.write(
            f"# {idx},{role},{layer.material},{layer.thickness_m:.12g},{thickness_um:.12g},"
            f"{alpha:.12g},{beta:.12g},{gamma:.12g}\n"
        )
    buffer.write(",".join(columns) + "\n")
    np.savetxt(buffer, data, delimiter=",", fmt="%.12g")
    return buffer.getvalue().encode("utf-8")


def _slug_token(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned.lower() or "layer"


def _stack_plot_filename_stem(mode: str) -> str:
    stack = build_stack_from_session()
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


def _dispersion_export_bytes(im_rpp: np.ndarray, kxv_cm1: np.ndarray, wv_cm1: np.ndarray, workers: int, speed_name: str) -> bytes:
    kx_grid, w_grid = np.meshgrid(np.asarray(kxv_cm1, dtype=float), np.asarray(wv_cm1, dtype=float), indexing="xy")
    data = np.column_stack(
        [
            np.asarray(im_rpp, dtype=float).reshape(-1),
            kx_grid.reshape(-1),
            w_grid.reshape(-1),
        ]
    )
    stack = build_stack_from_session()
    return _build_export_txt(
        mode="dispersion",
        stack=stack,
        data=data,
        columns=["rpp", "kx", "w"],
        workers=workers,
        preset_name=speed_name,
    )


def _isofrequency_export_bytes(
    im_rpp: np.ndarray,
    kxv_cm1: np.ndarray,
    phiv_rad: np.ndarray,
    w0_cm1: float,
    workers: int,
    speed_name: str,
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
    stack = build_stack_from_session()
    return _build_export_txt(
        mode="isofrequency",
        stack=stack,
        data=data,
        columns=["rpp", "kx", "w", "phi"],
        workers=workers,
        preset_name=speed_name,
    )


def _dispersion_plot_png_bytes(im_rpp: np.ndarray, kxv_cm1: np.ndarray, wv_cm1: np.ndarray) -> bytes:
    fig = plot_heatmap(
        x=np.asarray(kx_cm1_to_ui(kxv_cm1), dtype=float),
        y=np.asarray(wv_cm1, dtype=float),
        z=np.asarray(im_rpp, dtype=float),
        xlabel="kx (10^3 cm⁻¹)",
        ylabel="w (cm⁻¹)",
        title="Im(rpp) map",
        cmap=str(PLOT_COLORMAPS[str(st.session_state.get("plot_colormap", "Magma"))]).lower(),
    )
    return _figure_png_bytes(fig)


def _isofrequency_plot_png_bytes(im_rpp: np.ndarray, kxv_cm1: np.ndarray, phiv_rad: np.ndarray) -> bytes:
    fig = plot_polar_isofrequency(
        phi_rad=np.asarray(phiv_rad, dtype=float),
        kx=np.asarray(kx_cm1_to_ui(kxv_cm1), dtype=float),
        z=np.asarray(im_rpp, dtype=float),
        title="Isofrequency Im(rpp) polar surface",
        cmap=str(PLOT_COLORMAPS[str(st.session_state.get("plot_colormap", "Magma"))]).lower(),
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


def validate_iso_inputs(w0: float, kx_min: float, kx_max: float, nk: int, nphi: int) -> List[str]:
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
    grid_points = int(nk) * int(nphi)
    if grid_points > MAX_ISO_GRID_POINTS:
        errors.append(f"Isofrequency grid is too dense ({grid_points:,} points). Keep it below {MAX_ISO_GRID_POINTS:,}.")
    return errors


def _render_validation_errors(errors: List[str]) -> None:
    for message in errors:
        st.caption(f":red[• {message}]")


def _sync_preset_defaults(preset_name: str, speed_presets: Mapping[str, Mapping[str, int]]) -> None:
    if st.session_state.get("last_applied_speed_preset") == preset_name:
        return
    preset = speed_presets[preset_name]
    st.session_state.map_nw = int(preset["nw"])
    st.session_state.map_nk = int(preset["nk"])
    st.session_state.iso_nk = int(preset["nk"])
    st.session_state.iso_nphi = int(preset["nphi"])
    st.session_state.last_applied_speed_preset = preset_name


def _apply_selected_speed_preset(speed_presets: Mapping[str, Mapping[str, int]]) -> None:
    preset_name = str(st.session_state.get("speed_preset_choice", "Normal"))
    _sync_preset_defaults(preset_name, speed_presets)


def _current_calc_settings() -> Dict[str, Any]:
    mode = str(st.session_state.get("calc_mode", "Im(rpp) as f(w, kx)"))
    settings = {
        "mode": mode,
        "speed_name": str(st.session_state.get("speed_preset_choice", "Normal")),
        "workers": int(st.session_state.get("worker_count", 4)),
        "fast_preview_plots": bool(st.session_state.get("fast_preview_plots", True)),
        "advanced_euler_enabled": bool(st.session_state.get("advanced_euler_enabled", False)),
    }
    if mode == "Im(rpp) as f(w, kx)":
        settings["params"] = {
            "w_min": float(st.session_state.get("map_w_min", 780.0)),
            "w_max": float(st.session_state.get("map_w_max", 1050.0)),
            "nw": int(st.session_state.get("map_nw", 160)),
            "kx_min": float(st.session_state.get("map_kx_min", 0.05)),
            "kx_max": float(st.session_state.get("map_kx_max", 15.0)),
            "nk": int(st.session_state.get("map_nk", 180)),
        }
    else:
        settings["params"] = {
            "w0": float(st.session_state.get("iso_w0", 500.0)),
            "nk": int(st.session_state.get("iso_nk", 180)),
            "nphi": int(st.session_state.get("iso_nphi", 180)),
            "kx_min": float(st.session_state.get("iso_kx_min", 0.05)),
            "kx_max": float(st.session_state.get("iso_kx_max", 15.0)),
        }
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
        "preset": str(st.session_state.get("speed_preset_choice", "Normal")),
        "workers": str(st.session_state.get("worker_count", 4)),
        "mode": mode,
        "compute_state": compute_state,
        "freshness": freshness,
    }


def _store_compute_snapshot(mode: str) -> None:
    st.session_state.last_compute_signature = _current_input_signature()
    st.session_state.last_compute_mode = mode
    st.session_state.last_compute_timestamp = datetime.now(timezone.utc).isoformat()
    st.session_state.compute_state = "Computed"
    st.session_state.calc_settings = {
        "map": {
            "w_min": float(st.session_state.get("map_w_min", 780.0)),
            "w_max": float(st.session_state.get("map_w_max", 1050.0)),
            "nw": int(st.session_state.get("map_nw", 160)),
            "kx_min": float(st.session_state.get("map_kx_min", 0.05)),
            "kx_max": float(st.session_state.get("map_kx_max", 15.0)),
            "nk": int(st.session_state.get("map_nk", 180)),
        },
        "iso": {
            "w0": float(st.session_state.get("iso_w0", 500.0)),
            "nk": int(st.session_state.get("iso_nk", 180)),
            "nphi": int(st.session_state.get("iso_nphi", 180)),
            "kx_min": float(st.session_state.get("iso_kx_min", 0.05)),
            "kx_max": float(st.session_state.get("iso_kx_max", 15.0)),
        },
    }


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
    status = st.empty()
    progress = st.progress(0.0)

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


def _peak_threshold_fraction() -> float:
    threshold_percent = float(st.session_state.get("peak_dot_threshold_percent", 10.0))
    return float(np.clip(threshold_percent, 0.0, 100.0)) / 100.0


def _peak_overlay_for_map(kx_ui: np.ndarray, wv: np.ndarray, im_rpp: np.ndarray) -> dict[str, np.ndarray] | None:
    if not bool(st.session_state.get("show_peak_dots", True)):
        return None
    values = np.asarray(im_rpp, dtype=float)
    if values.size == 0:
        return None
    global_max = float(np.max(values))
    if global_max <= 0.0:
        return None
    threshold = _peak_threshold_fraction() * global_max
    row_max = np.max(values, axis=1)
    keep = row_max >= threshold
    if not np.any(keep):
        return None
    peak_idx = np.argmax(values, axis=1)
    rows = np.nonzero(keep)[0]
    return {
        "x": np.asarray(kx_ui[peak_idx[rows]], dtype=float),
        "y": np.asarray(wv[rows], dtype=float),
        "value": np.asarray(row_max[rows], dtype=float),
    }


def _peak_overlay_for_iso(phi_rad: np.ndarray, kx_ui: np.ndarray, im_rpp: np.ndarray) -> dict[str, np.ndarray] | None:
    if not bool(st.session_state.get("show_peak_dots", True)):
        return None
    values = np.asarray(im_rpp, dtype=float)
    if values.size == 0:
        return None
    global_max = float(np.max(values))
    if global_max <= 0.0:
        return None
    threshold = _peak_threshold_fraction() * global_max
    row_max = np.max(values, axis=1)
    keep = row_max >= threshold
    if not np.any(keep):
        return None
    peak_idx = np.argmax(values, axis=1)
    rows = np.nonzero(keep)[0]
    return {
        "theta_deg": np.rad2deg(np.asarray(phi_rad[rows], dtype=float)),
        "r": np.asarray(kx_ui[peak_idx[rows]], dtype=float),
        "value": np.asarray(row_max[rows], dtype=float),
    }


def _render_map_compute_form(speed_name: str, workers: int) -> None:
    with st.form("map_form", border=False):
        with st.container():
            st.markdown('<p class="section-label">Range and sampling</p>', unsafe_allow_html=True)
            freq_cols = st.columns(2, gap="small")
            freq_cols[0].number_input("w min (cm⁻¹)", step=10.0, key="map_w_min")
            freq_cols[1].number_input("w max (cm⁻¹)", step=10.0, key="map_w_max")

            mom_cols = st.columns(2, gap="small")
            mom_cols[0].number_input("kx min (10^3 cm⁻¹)", step=0.05, format="%.3f", key="map_kx_min")
            mom_cols[1].number_input("kx max (10^3 cm⁻¹)", step=0.1, format="%.3f", key="map_kx_max")
            sample_cols = st.columns(2, gap="small")
            sample_cols[0].number_input("Nw", min_value=8, step=8, key="map_nw")
            sample_cols[1].number_input("Nk", min_value=8, step=8, key="map_nk")

        errors = validate_map_inputs(
            w_min=float(st.session_state.map_w_min),
            w_max=float(st.session_state.map_w_max),
            nw=int(st.session_state.map_nw),
            kx_min=float(st.session_state.map_kx_min),
            kx_max=float(st.session_state.map_kx_max),
            nk=int(st.session_state.map_nk),
        )
        grid_points = int(st.session_state.map_nw) * int(st.session_state.map_nk)
        with st.container():
            st.markdown('<p class="section-label">Compute summary</p>', unsafe_allow_html=True)
            st.caption(f"Grid: {grid_points:,} points")
            st.caption(f"Cost: {_estimated_cost_label(grid_points, workers)} on {workers} workers")
            _render_validation_errors(errors)
            run_map = st.form_submit_button(
                ":material/play_arrow: Compute Im(rpp) map",
                width="stretch",
                type="primary",
                disabled=bool(errors),
            )

    if run_map:
        st.session_state.compute_state = "Running"
        stack = build_stack_from_session()
        wv, kxv, im_rpp = _run_with_progress(
            start_message="Preparing dispersion computation...",
            complete_message="Dispersion computation complete",
            compute_fn=lambda progress_cb: compute_rpp_map(
                stack,
                w_min=float(st.session_state.map_w_min),
                w_max=float(st.session_state.map_w_max),
                nw=int(st.session_state.map_nw),
                kx_min=float(kx_ui_to_cm1(st.session_state.map_kx_min)),
                kx_max=float(kx_ui_to_cm1(st.session_state.map_kx_max)),
                nk=int(st.session_state.map_nk),
                workers=int(workers),
                progress=progress_cb,
            ),
        )
        st.session_state.map_result = (wv, kxv, im_rpp)
        _store_compute_snapshot("Im(rpp) as f(w, kx)")


def _render_iso_compute_form(speed_name: str, workers: int) -> None:
    with st.form("iso_form", border=False):
        with st.container():
            st.markdown('<p class="section-label">Range and sampling</p>', unsafe_allow_html=True)
            freq_cols = st.columns(3, gap="small")
            freq_cols[0].number_input("w0 (cm⁻¹)", step=5.0, key="iso_w0")
            freq_cols[1].number_input("Nk", min_value=8, step=8, key="iso_nk")
            freq_cols[2].number_input("Nphi", min_value=8, step=8, key="iso_nphi")

            mom_cols = st.columns(2, gap="small")
            mom_cols[0].number_input("kx min (10^3 cm⁻¹)", step=0.05, format="%.3f", key="iso_kx_min")
            mom_cols[1].number_input("kx max (10^3 cm⁻¹)", step=0.1, format="%.3f", key="iso_kx_max")

        errors = validate_iso_inputs(
            w0=float(st.session_state.iso_w0),
            kx_min=float(st.session_state.iso_kx_min),
            kx_max=float(st.session_state.iso_kx_max),
            nk=int(st.session_state.iso_nk),
            nphi=int(st.session_state.iso_nphi),
        )
        grid_points = int(st.session_state.iso_nk) * int(st.session_state.iso_nphi)
        with st.container():
            st.markdown('<p class="section-label">Compute summary</p>', unsafe_allow_html=True)
            st.caption(f"Grid: {grid_points:,} points")
            st.caption(f"Cost: {_estimated_cost_label(grid_points, workers)} on {workers} workers")
            _render_validation_errors(errors)
            run_iso = st.form_submit_button(
                ":material/play_arrow: Compute isofrequency Im(rpp)",
                width="stretch",
                type="primary",
                disabled=bool(errors),
            )

    if run_iso:
        st.session_state.compute_state = "Running"
        stack = build_stack_from_session()
        phiv, kxv, im_rpp = _run_with_progress(
            start_message="Preparing isofrequency computation...",
            complete_message="Isofrequency computation complete",
            compute_fn=lambda progress_cb: compute_isofreq_map(
                stack,
                w0=float(st.session_state.iso_w0),
                kx_min=float(kx_ui_to_cm1(st.session_state.iso_kx_min)),
                kx_max=float(kx_ui_to_cm1(st.session_state.iso_kx_max)),
                nk=int(st.session_state.iso_nk),
                nphi=int(st.session_state.iso_nphi),
                global_phi_sweep=True,
                workers=int(workers),
                progress=progress_cb,
            ),
        )
        st.session_state.iso_result = (phiv, kxv, im_rpp)
        _store_compute_snapshot("Isofrequency surface")


def render_calculation_panel(speed_presets: Mapping[str, Mapping[str, int]]) -> None:
    """Render the calculation-setup column for both supported computation modes."""
    speed_name = str(st.session_state.get("speed_preset_choice", "Normal"))
    workers = int(st.session_state.get("worker_count", 4))

    st.markdown('<p class="section-label">Calculation setup</p>', unsafe_allow_html=True)
    st.subheader("Run controls", anchor=False)

    # Group global run context separately so mode-specific forms stay compact and explicit.
    with st.container():
        st.markdown('<p class="section-label">Run context</p>', unsafe_allow_html=True)
        single_choice(
            "Calculation mode",
            ["Im(rpp) as f(w, kx)", "Isofrequency surface"],
            str(st.session_state.get("calc_mode", "Im(rpp) as f(w, kx)")),
            "calc_mode",
        )
        ctx_cols = st.columns(2, gap="small")
        ctx_cols[0].selectbox(
            "Preset",
            list(speed_presets.keys()),
            key="speed_preset_choice",
            on_change=_apply_selected_speed_preset,
            args=(speed_presets,),
        )
        ctx_cols[1].caption(f"Workers: {int(st.session_state.get('worker_count', 4))}")
        st.toggle("Preview plots", key="fast_preview_plots", help="Downsamples only the displayed plot. Computation stays full resolution.")
        with st.expander("Advanced", expanded=bool(st.session_state.get("advanced_euler_enabled", False)), icon=":material/tune:"):
            st.slider("CPU workers", min_value=1, max_value=16, step=1, key="worker_count")
            st.toggle("Enable advanced Euler (α/β/γ)", key="advanced_euler_enabled")

    speed_name = str(st.session_state.get("speed_preset_choice", "Normal"))
    workers = int(st.session_state.get("worker_count", 4))
    if st.session_state.calc_mode == "Im(rpp) as f(w, kx)":
        _render_map_compute_form(speed_name=speed_name, workers=workers)
    else:
        _render_iso_compute_form(speed_name=speed_name, workers=workers)

    # Mirror the live calculation UI so unrelated reruns can restore the current mode and sampling values.
    st.session_state["_calc_ui_state_mirror"] = _calc_ui_state_snapshot()


def _render_plot_toolbar(
    has_result: bool,
    export_bytes: bytes | None,
    file_name: str | None,
    image_bytes: bytes | None,
    image_file_name: str | None,
) -> None:
    with st.container():
        toolbar_cols = st.columns([0.2, 0.16, 0.12, 0.12, 0.18, 0.11, 0.11], gap="small", vertical_alignment="center")
        with toolbar_cols[0]:
            st.selectbox("Colormap", list(PLOT_COLORMAPS.keys()), key="plot_colormap")
        with toolbar_cols[1]:
            st.write("")
            if st.button(":material/refresh: Reset view", width="stretch", disabled=not has_result):
                st.session_state.plot_refresh_nonce = int(st.session_state.get("plot_refresh_nonce", 0)) + 1
        with toolbar_cols[2]:
            st.toggle("Peak dots", key="show_peak_dots", disabled=not has_result)
        with toolbar_cols[3]:
            st.number_input(
                "Threshold %",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key="peak_dot_threshold_percent",
                disabled=not has_result or not bool(st.session_state.get("show_peak_dots", True)),
            )
        with toolbar_cols[4]:
            freshness = "Current" if _results_are_current() else "Stale"
            st.markdown(
                f'<div style="padding-top:0.35rem;"><span class="status-pill {"current" if freshness == "Current" else "stale"}">{freshness}</span></div>',
                unsafe_allow_html=True,
            )
        with toolbar_cols[5]:
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
        with toolbar_cols[6]:
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


def _current_export_payload(speed_name: str, workers: int) -> tuple[bytes | None, str | None]:
    mode = str(st.session_state.get("calc_mode", "Im(rpp) as f(w, kx)"))
    if mode == "Im(rpp) as f(w, kx)" and st.session_state.map_result is not None:
        wv, kxv, im_rpp = st.session_state.map_result
        return (
            _dispersion_export_bytes(im_rpp=im_rpp, kxv_cm1=kxv, wv_cm1=wv, workers=workers, speed_name=speed_name),
            f"{_stack_plot_filename_stem('dispersion-rpp-export')}.txt",
        )
    if mode == "Isofrequency surface" and st.session_state.iso_result is not None:
        phiv, kxv, im_rpp = st.session_state.iso_result
        return (
            _isofrequency_export_bytes(
                im_rpp=im_rpp,
                kxv_cm1=kxv,
                phiv_rad=phiv,
                w0_cm1=float(st.session_state.get("iso_w0", 500.0)),
                workers=workers,
                speed_name=speed_name,
            ),
            f"{_stack_plot_filename_stem('isofrequency-rpp-export')}.txt",
        )
    return None, None


def _current_plot_image_payload() -> tuple[bytes | None, str | None]:
    mode = str(st.session_state.get("calc_mode", "Im(rpp) as f(w, kx)"))
    if mode == "Im(rpp) as f(w, kx)" and st.session_state.map_result is not None:
        wv, kxv, im_rpp = st.session_state.map_result
        return (
            _dispersion_plot_png_bytes(im_rpp=im_rpp, kxv_cm1=kxv, wv_cm1=wv),
            f"{_stack_plot_filename_stem('im-rpp-map')}.png",
        )
    if mode == "Isofrequency surface" and st.session_state.iso_result is not None:
        phiv, kxv, im_rpp = st.session_state.iso_result
        return (
            _isofrequency_plot_png_bytes(im_rpp=im_rpp, kxv_cm1=kxv, phiv_rad=phiv),
            f"{_stack_plot_filename_stem('isofrequency-im-rpp')}.png",
        )
    return None, None


def _render_map_plot(speed_name: str, workers: int) -> bytes | None:
    if st.session_state.map_result is None:
        st.caption(":material/analytics: Run the computation to show the map.")
        return None
    wv, kxv, im_rpp = st.session_state.map_result
    full_kx_ui = np.asarray(kx_cm1_to_ui(kxv), dtype=float)
    full_w = np.asarray(wv, dtype=float)
    full_z = np.asarray(im_rpp, dtype=float)
    peak_overlay = _peak_overlay_for_map(full_kx_ui, full_w, full_z)
    fast_preview = bool(st.session_state.get("fast_preview_plots", True))
    if fast_preview:
        kx_plot, w_plot, z_plot = _downsample_grid_for_preview(
            full_kx_ui,
            full_w,
            full_z,
            max_x=FAST_PREVIEW_MAX_X,
            max_y=FAST_PREVIEW_MAX_Y,
        )
        st.caption(f"Preview: {z_plot.shape[0]}x{z_plot.shape[1]} shown from {im_rpp.shape[0]}x{im_rpp.shape[1]}.")
    else:
        kx_plot = full_kx_ui
        w_plot = full_w
        z_plot = full_z
    fig = plot_heatmap_interactive(
        kx_plot,
        w_plot,
        z_plot,
        xlabel="kx (10^3 cm⁻¹)",
        ylabel="w (cm⁻¹)",
        title="Im(rpp) map",
        cmap=PLOT_COLORMAPS[str(st.session_state.get("plot_colormap", "Magma"))],
        height=455,
        peak_overlay=peak_overlay,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        config=PLOTLY_CONFIG,
        key=f"map_plot_{st.session_state.get('plot_refresh_nonce', 0)}",
    )
    return _dispersion_export_bytes(im_rpp=im_rpp, kxv_cm1=kxv, wv_cm1=wv, workers=workers, speed_name=speed_name)


def _render_iso_plot(speed_name: str, workers: int) -> bytes | None:
    if st.session_state.iso_result is None:
        st.caption(":material/radar: Run the computation to show the surface.")
        return None
    phiv, kxv, im_rpp = st.session_state.iso_result
    full_phi = np.asarray(phiv, dtype=float)
    full_kx_ui = np.asarray(kx_cm1_to_ui(kxv), dtype=float)
    full_z = np.asarray(im_rpp, dtype=float)
    peak_overlay = _peak_overlay_for_iso(full_phi, full_kx_ui, full_z)
    fast_preview = bool(st.session_state.get("fast_preview_plots", True))
    if fast_preview:
        kx_plot, phi_plot, z_plot = _downsample_grid_for_preview(
            full_kx_ui,
            full_phi,
            full_z,
            max_x=FAST_PREVIEW_MAX_R,
            max_y=FAST_PREVIEW_MAX_PHI,
        )
        st.caption(f"Preview: {z_plot.shape[0]}x{z_plot.shape[1]} shown from {im_rpp.shape[0]}x{im_rpp.shape[1]}.")
    else:
        phi_plot = full_phi
        kx_plot = full_kx_ui
        z_plot = full_z
    fig = plot_polar_isofrequency_interactive(
        phi_plot,
        kx_plot,
        z_plot,
        title="Isofrequency Im(rpp) polar surface",
        radial_label="kx (10^3 cm^-1)",
        cmap=PLOT_COLORMAPS[str(st.session_state.get("plot_colormap", "Magma"))],
        height=455,
        peak_overlay=peak_overlay,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        config=PLOTLY_CONFIG,
        key=f"iso_plot_{st.session_state.get('plot_refresh_nonce', 0)}",
    )
    return _isofrequency_export_bytes(
        im_rpp=im_rpp,
        kxv_cm1=kxv,
        phiv_rad=phiv,
        w0_cm1=float(st.session_state.get("iso_w0", 500.0)),
        workers=workers,
        speed_name=speed_name,
    )


def _render_stack_preview_tab() -> None:
    preview_stack = build_stack_from_session()
    figure = plot_stack_pseudo3d(preview_stack)
    st.pyplot(figure, width="stretch", clear_figure=True)
    plt.close(figure)


def _render_metadata_tab() -> None:
    settings = _current_calc_settings()
    st.markdown("**Run context**")
    st.json(settings, expanded=False)
    st.markdown("**Snapshot**")
    st.json(
        {
            "compute_state": st.session_state.get("compute_state", "Idle"),
            "last_compute_mode": st.session_state.get("last_compute_mode"),
            "last_compute_timestamp": st.session_state.get("last_compute_timestamp"),
            "freshness": "Current" if _results_are_current() else "Stale",
        },
        expanded=False,
    )


def _render_export_tab(speed_name: str, workers: int) -> None:
    mode = str(st.session_state.get("calc_mode", "Im(rpp) as f(w, kx)"))
    if mode == "Im(rpp) as f(w, kx)" and st.session_state.map_result is not None:
        wv, kxv, im_rpp = st.session_state.map_result
        export_bytes = _dispersion_export_bytes(im_rpp=im_rpp, kxv_cm1=kxv, wv_cm1=wv, workers=workers, speed_name=speed_name)
        image_bytes = _dispersion_plot_png_bytes(im_rpp=im_rpp, kxv_cm1=kxv, wv_cm1=wv)
        st.download_button(
            "Download dispersion export",
            data=export_bytes,
            file_name=f"{_stack_plot_filename_stem('dispersion-rpp-export')}.txt",
            mime="text/plain",
            width="stretch",
        )
        st.download_button(
            "Download plot image",
            data=image_bytes,
            file_name=f"{_stack_plot_filename_stem('im-rpp-map')}.png",
            mime="image/png",
            width="stretch",
        )
    elif mode == "Isofrequency surface" and st.session_state.iso_result is not None:
        phiv, kxv, im_rpp = st.session_state.iso_result
        export_bytes = _isofrequency_export_bytes(
            im_rpp=im_rpp,
            kxv_cm1=kxv,
            phiv_rad=phiv,
            w0_cm1=float(st.session_state.get("iso_w0", 500.0)),
            workers=workers,
            speed_name=speed_name,
        )
        image_bytes = _isofrequency_plot_png_bytes(im_rpp=im_rpp, kxv_cm1=kxv, phiv_rad=phiv)
        st.download_button(
            "Download isofrequency export",
            data=export_bytes,
            file_name=f"{_stack_plot_filename_stem('isofrequency-rpp-export')}.txt",
            mime="text/plain",
            width="stretch",
        )
        st.download_button(
            "Download plot image",
            data=image_bytes,
            file_name=f"{_stack_plot_filename_stem('isofrequency-im-rpp')}.png",
            mime="image/png",
            width="stretch",
        )
    else:
        st.caption("No computed result is available for export yet.")


def render_results_panel() -> None:
    """Render the main plot, status tabs, and export actions."""
    speed_name = str(st.session_state.get("speed_preset_choice", "Normal"))
    workers = int(st.session_state.get("worker_count", 4))
    mode = str(st.session_state.get("calc_mode", "Im(rpp) as f(w, kx)"))

    st.markdown('<p class="section-label">Results</p>', unsafe_allow_html=True)
    st.subheader("Main plot", anchor=False)

    with st.container():
        has_result = (mode == "Im(rpp) as f(w, kx)" and st.session_state.map_result is not None) or (
            mode == "Isofrequency surface" and st.session_state.iso_result is not None
        )
        export_bytes, export_name = _current_export_payload(speed_name=speed_name, workers=workers)
        image_bytes, image_name = _current_plot_image_payload()
        _render_plot_toolbar(
            has_result=has_result,
            export_bytes=export_bytes,
            file_name=export_name,
            image_bytes=image_bytes,
            image_file_name=image_name,
        )
    with st.container(border=True):
        if mode == "Im(rpp) as f(w, kx)":
            export_bytes = _render_map_plot(speed_name=speed_name, workers=workers)
        else:
            export_bytes = _render_iso_plot(speed_name=speed_name, workers=workers)

    tab_result, tab_stack, tab_meta, tab_export = st.tabs(["Status", "Preview", "Metadata", "Export"])
    with tab_result:
        st.markdown(
            f"**Displayed mode:** {'Im(rpp) map' if mode == 'Im(rpp) as f(w, kx)' else 'Isofrequency surface'}  \n"
            f"**Plot freshness:** {'Current' if _results_are_current() else 'Stale'}"
        )
        if st.session_state.get("last_compute_timestamp"):
            st.caption(f"Last compute timestamp (UTC): {st.session_state['last_compute_timestamp']}")
    with tab_stack:
        _render_stack_preview_tab()
    with tab_meta:
        _render_metadata_tab()
    with tab_export:
        _render_export_tab(speed_name=speed_name, workers=workers)
