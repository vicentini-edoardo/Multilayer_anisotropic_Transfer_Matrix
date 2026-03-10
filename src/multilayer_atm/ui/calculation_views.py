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

from ..custom_materials import material_frequency_range_warnings
from ..plotting import (
    plot_heatmap,
    plot_heatmap_interactive,
    plot_polar_isofrequency,
    plot_polar_isofrequency_interactive,
)
from ..solver import compute_isofreq_map, compute_rpp_map
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
    "show_peak_dots",
    "peak_dot_threshold_percent",
    "map_state",
    "iso_state",
)

def _initial_calc_defaults(speed_presets: Mapping[str, Mapping[str, Mapping[str, int]]]) -> Dict[str, Any]:
    default_preset = "Normal" if "Normal" in speed_presets else next(iter(speed_presets))
    return {
        "calc_mode": "Im(rpp) as f(w, kx)",
        "map_resolution_choice": default_preset,
        "iso_resolution_choice": default_preset,
        "worker_count": 4,
        "fast_preview_plots": True,
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


def _dispersion_export_bytes(im_rpp: np.ndarray, kxv_cm1: np.ndarray, wv_cm1: np.ndarray, workers: int, resolution_name: str) -> bytes:
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
        resolution_name=resolution_name,
    )


def _isofrequency_export_bytes(
    im_rpp: np.ndarray,
    kxv_cm1: np.ndarray,
    phiv_rad: np.ndarray,
    w0_cm1: float,
    workers: int,
    resolution_name: str,
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
        resolution_name=resolution_name,
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
        "workers": int(st.session_state.get("worker_count", 4)),
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


def _append_map_history(wv: np.ndarray, kxv: np.ndarray, im_rpp: np.ndarray) -> None:
    entry = {
        "id": _next_history_id("map"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "resolution": str(st.session_state.get("map_resolution_choice", "Normal")),
        "stack_summary": _stack_history_summary("map"),
        "payload": (np.asarray(wv, dtype=float), np.asarray(kxv, dtype=float), np.asarray(im_rpp, dtype=float)),
    }
    st.session_state.map_history.append(entry)
    st.session_state.selected_map_history_id = str(entry["id"])


def _append_iso_history(phiv: np.ndarray, kxv: np.ndarray, im_rpp: np.ndarray) -> None:
    entry = {
        "id": _next_history_id("iso"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "resolution": str(st.session_state.get("iso_resolution_choice", "Normal")),
        "stack_summary": _stack_history_summary("iso"),
        "payload": (np.asarray(phiv, dtype=float), np.asarray(kxv, dtype=float), np.asarray(im_rpp, dtype=float)),
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
        "Saved map",
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


def _render_map_input_strip(speed_presets: Mapping[str, Mapping[str, Mapping[str, int]]]) -> None:
    map_state = _mode_state("map")
    map_custom = str(st.session_state.get("map_resolution_choice", "Normal")) == "Custom"

    top_cols = st.columns(3, gap=None)
    with top_cols[0]:
        _mode_number_input("map", "phi0", "phi0 (deg)", step=1.0, format="%.1f")
    with top_cols[2]:
        st.selectbox("Resolution", _resolution_choices(speed_presets), key="map_resolution_choice")

    freq_cols = st.columns(3, gap=None)
    with freq_cols[0]:
        _mode_number_input("map", "w_min", "w min (cm⁻¹)", step=10.0)
    with freq_cols[1]:
        _mode_number_input("map", "w_max", "w max (cm⁻¹)", step=10.0)
    with freq_cols[2]:
        _mode_number_input("map", "nw", "Nw", min_value=8, step=8, disabled=not map_custom)

    mom_cols = st.columns(3, gap=None)
    with mom_cols[0]:
        _mode_number_input("map", "kx_min", "kx min (10^3 cm⁻¹)", step=0.05, format="%.3f")
    with mom_cols[1]:
        _mode_number_input("map", "kx_max", "kx max (10^3 cm⁻¹)", step=0.1, format="%.3f")
    with mom_cols[2]:
        _mode_number_input("map", "nk", "Nk", min_value=8, step=8, disabled=not map_custom)


def _render_iso_input_strip(speed_presets: Mapping[str, Mapping[str, Mapping[str, int]]]) -> None:
    iso_state = _mode_state("iso")
    iso_custom = str(st.session_state.get("iso_resolution_choice", "Normal")) == "Custom"

    top_cols = st.columns([1.0, 1.0, 1.0], gap=None)
    with top_cols[0]:
        _mode_number_input("iso", "w0", "w0 (cm⁻¹)", step=5.0)
    with top_cols[2]:
        st.selectbox("Resolution", _resolution_choices(speed_presets), key="iso_resolution_choice")

    phi_cols = st.columns(3, gap=None)
    with phi_cols[0]:
        _mode_number_input("iso", "phi_min_deg", "phi0 min (deg)", step=1.0, format="%.1f")
    with phi_cols[1]:
        _mode_number_input("iso", "phi_max_deg", "phi0 max (deg)", step=1.0, format="%.1f")
    with phi_cols[2]:
        _mode_number_input("iso", "nphi", "Nphi0", min_value=8, step=8, disabled=not iso_custom)

    mom_cols = st.columns(3, gap=None)
    with mom_cols[0]:
        _mode_number_input("iso", "kx_min", "kx min (10^3 cm⁻¹)", step=0.05, format="%.3f")
    with mom_cols[1]:
        _mode_number_input("iso", "kx_max", "kx max (10^3 cm⁻¹)", step=0.1, format="%.3f")
    with mom_cols[2]:
        _mode_number_input("iso", "nk", "Nk", min_value=8, step=8, disabled=not iso_custom)


def _active_compute_summary() -> tuple[List[str], List[str], int, str]:
    mode = str(st.session_state.get("calc_mode", "Im(rpp) as f(w, kx)"))
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
    mode = str(st.session_state.get("calc_mode", "Im(rpp) as f(w, kx)"))
    st.session_state.compute_state = "Running"
    stack = build_stack_from_session()
    if mode == "Im(rpp) as f(w, kx)":
        map_state = _mode_state("map")
        stack_for_run = stack.with_interior_alpha_offset(float(map_state.get("phi0", 0.0)))
        wv, kxv, im_rpp = _run_with_progress(
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
            ),
        )
        _append_map_history(wv, kxv, im_rpp)
        _sync_selected_history_result("map")
        _store_compute_snapshot("Im(rpp) as f(w, kx)")
        return

    iso_state = _mode_state("iso")
    phiv, kxv, im_rpp = _run_with_progress(
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
        ),
    )
    _append_iso_history(phiv, kxv, im_rpp)
    _sync_selected_history_result("iso")
    _store_compute_snapshot("Isofrequency surface")


def render_run_controls_panel(speed_presets: Mapping[str, Mapping[str, Mapping[str, int]]]) -> None:
    """Render the run controls card with mode, compute summary, and compute action."""
    _sync_resolution_defaults(speed_presets)
    if hasattr(st, "segmented_control"):
        st.segmented_control(
            "Calculation mode",
            ["Im(rpp) as f(w, kx)", "Isofrequency surface"],
            key="calc_mode",
        )
    else:
        st.radio(
            "Calculation mode",
            ["Im(rpp) as f(w, kx)", "Isofrequency surface"],
            horizontal=True,
            key="calc_mode",
        )
    workers = int(st.session_state.get("worker_count", 4))
    errors, warnings, _, button_label = _active_compute_summary()
    for warning in warnings:
        st.warning(warning, icon=":material/warning:")
    _render_validation_errors(errors)
    with st.form("run_controls_form", border=False):
        button_col, _ = st.columns([0.72, 0.28], gap=None)
        with button_col:
            run_now = st.form_submit_button(button_label, width="stretch", type="primary", disabled=bool(errors))
    if run_now:
        _execute_active_compute(workers=workers)


def render_mode_input_strip(speed_presets: Mapping[str, Mapping[str, Mapping[str, int]]]) -> None:
    """Render only the active mode-specific sampling inputs."""
    _sync_resolution_defaults(speed_presets)
    if st.session_state.get("calc_mode") == "Im(rpp) as f(w, kx)":
        _render_map_input_strip(speed_presets=speed_presets)
    else:
        _render_iso_input_strip(speed_presets=speed_presets)


def _render_plot_toolbar(
    has_result: bool,
    export_bytes: bytes | None,
    file_name: str | None,
    image_bytes: bytes | None,
    image_file_name: str | None,
) -> None:
    with st.container():
        st.caption(":material/tune: Plot controls")
        settings_row = st.columns([0.30, 0.18, 0.18, 0.16, 0.18], gap=None, vertical_alignment="bottom")
        with settings_row[0]:
            st.selectbox("Colormap", list(PLOT_COLORMAPS.keys()), key="plot_colormap")
        with settings_row[1]:
            st.toggle(
                "Preview plots",
                key="fast_preview_plots",
                help="Downsamples only the displayed plot. Computation stays full resolution.",
            )
        with settings_row[2]:
            st.toggle("Peak dots", key="show_peak_dots", disabled=not has_result)
        with settings_row[3]:
            st.number_input(
                "Threshold %",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key="peak_dot_threshold_percent",
                disabled=not has_result or not bool(st.session_state.get("show_peak_dots", True)),
            )
        with settings_row[4]:
            freshness = "Current" if _results_are_current() else "Stale"
            st.caption(f":material/monitoring: State: {freshness}")

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
    if mode == "Im(rpp) as f(w, kx)" and st.session_state.map_result is not None:
        wv, kxv, im_rpp = st.session_state.map_result
        return (
            _dispersion_export_bytes(im_rpp=im_rpp, kxv_cm1=kxv, wv_cm1=wv, workers=workers, resolution_name=resolution_name),
            f"{_stack_plot_filename_stem('dispersion-rpp-export')}.txt",
        )
    if mode == "Isofrequency surface" and st.session_state.iso_result is not None:
        phiv, kxv, im_rpp = st.session_state.iso_result
        return (
            _isofrequency_export_bytes(
                im_rpp=im_rpp,
                kxv_cm1=kxv,
                phiv_rad=phiv,
                w0_cm1=float(_mode_state("iso")["w0"]),
                workers=workers,
                resolution_name=resolution_name,
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


def _render_map_plot(resolution_name: str, workers: int) -> bytes | None:
    _render_history_selector("map")
    if st.session_state.map_result is None:
        st.caption(":material/analytics: Run the computation to show the map.")
        return None
    wv, kxv, im_rpp = st.session_state.map_result
    full_kx_ui = np.asarray(kx_cm1_to_ui(kxv), dtype=float)
    full_w = np.asarray(wv, dtype=float)
    full_z = np.asarray(im_rpp, dtype=float)
    peak_overlay = _peak_overlay_for_map(full_kx_ui, full_w, full_z)
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
    if preview_caption:
        st.caption(preview_caption)
    return _dispersion_export_bytes(im_rpp=im_rpp, kxv_cm1=kxv, wv_cm1=wv, workers=workers, resolution_name=resolution_name)


def _render_iso_plot(resolution_name: str, workers: int) -> bytes | None:
    _render_history_selector("iso")
    if st.session_state.iso_result is None:
        st.caption(":material/radar: Run the computation to show the surface.")
        return None
    phiv, kxv, im_rpp = st.session_state.iso_result
    full_phi = np.asarray(phiv, dtype=float)
    full_kx_ui = np.asarray(kx_cm1_to_ui(kxv), dtype=float)
    full_z = np.asarray(im_rpp, dtype=float)
    peak_overlay = _peak_overlay_for_iso(full_phi, full_kx_ui, full_z)
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
    if preview_caption:
        st.caption(preview_caption)
    return _isofrequency_export_bytes(
        im_rpp=im_rpp,
        kxv_cm1=kxv,
        phiv_rad=phiv,
        w0_cm1=float(_mode_state("iso")["w0"]),
        workers=workers,
        resolution_name=resolution_name,
    )


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
    if mode == "Im(rpp) as f(w, kx)" and st.session_state.map_result is not None:
        wv, kxv, im_rpp = st.session_state.map_result
        export_bytes = _dispersion_export_bytes(im_rpp=im_rpp, kxv_cm1=kxv, wv_cm1=wv, workers=workers, resolution_name=resolution_name)
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
            w0_cm1=float(_mode_state("iso")["w0"]),
            workers=workers,
            resolution_name=resolution_name,
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
    resolution_name = _active_resolution_choice()
    workers = int(st.session_state.get("worker_count", 4))
    mode = str(st.session_state.get("calc_mode", "Im(rpp) as f(w, kx)"))
    _sync_selected_history_result("map" if mode == "Im(rpp) as f(w, kx)" else "iso")

    with st.container(gap=None):
        with st.container(border=True):
            if mode == "Im(rpp) as f(w, kx)":
                _render_map_plot(resolution_name=resolution_name, workers=workers)
            else:
                _render_iso_plot(resolution_name=resolution_name, workers=workers)

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
