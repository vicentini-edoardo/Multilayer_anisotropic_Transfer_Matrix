"""Stack construction and layer-editing controls for the Streamlit UI."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import streamlit as st

from ..models import DopingSpec, LayerSpec, StackSpec
from ..plotting import plot_stack_pseudo3d
from ..presets import default_layer_stack, example_layer_stack
from .material_builder import (
    custom_material_notes,
    handle_material_selection_change,
    init_custom_material_state,
    material_selector_options,
    material_selection_callback_args,
    maybe_open_custom_material_dialog,
)


LAYER_WIDGET_KEY_PREFIXES: Sequence[str] = (
    "sel_",
    "del_",
    "move_up_",
    "move_down_",
    "mat_",
    "thk_",
    "a_",
    "b_",
    "g_",
    "dop_",
    "wp_",
    "gp_",
)

CALC_STATE_KEYS_TO_PRESERVE: Sequence[str] = (
    "calc_mode",
    "map_resolution_choice",
    "iso_resolution_choice",
    "worker_count",
    "fast_preview_plots",
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

def _new_layer(material: str = "vac", thickness_m: float = 0.5e-6) -> Dict[str, object]:
    layer_id = f"layer_{st.session_state.layer_seq}"
    st.session_state.layer_seq += 1
    return {
        "id": layer_id,
        "material": material,
        "thickness_m": thickness_m,
        "alpha_rel_substrate_deg": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "gamma": 0.0,
        "doping_enabled": False,
        "wp_cm1": 0.0,
        "gp_cm1": 0.0,
    }


def init_layer_state() -> None:
    """Initialize per-session stack editor state."""
    init_custom_material_state()
    st.session_state.setdefault("layer_seq", 2)
    st.session_state.setdefault("layers", default_layer_stack())
    st.session_state.setdefault("selected_layer_id", None)
    for layer in st.session_state.layers:
        if "id" not in layer:
            layer["id"] = f"layer_{st.session_state.layer_seq}"
            st.session_state.layer_seq += 1
        layer.setdefault("alpha_rel_substrate_deg", float(layer.get("alpha", 0.0)))
        layer.setdefault("alpha", float(layer.get("alpha_rel_substrate_deg", 0.0)))
        layer.setdefault("beta", 0.0)
        layer.setdefault("gamma", 0.0)
        layer.setdefault("doping_enabled", False)
        layer.setdefault("wp_cm1", 0.0)
        layer.setdefault("gp_cm1", 0.0)

    _ensure_selected_layer()
    _cleanup_layer_widget_state(str(layer["id"]) for layer in st.session_state.layers)


def layer_widget_key(prefix: str, layer_id: str) -> str:
    return f"{prefix}{layer_id}"


def _init_layer_widget_state(layer_id: str, prefix: str, value: object) -> str:
    key = layer_widget_key(prefix, layer_id)
    st.session_state.setdefault(key, value)
    return key


def stale_layer_widget_keys(state_keys: Iterable[str], active_layer_ids: Iterable[str]) -> List[str]:
    active_ids = {str(layer_id) for layer_id in active_layer_ids}
    stale: List[str] = []
    for key in state_keys:
        for prefix in LAYER_WIDGET_KEY_PREFIXES:
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix) :]
            if suffix.isdigit() or suffix not in active_ids:
                stale.append(key)
            break
    return stale


def _cleanup_layer_widget_state(active_layer_ids: Iterable[str]) -> None:
    stale = stale_layer_widget_keys(st.session_state.keys(), active_layer_ids)
    for key in stale:
        st.session_state.pop(key, None)


def _preserve_calc_state_before_rerun() -> None:
    st.session_state["_preserved_calc_state"] = {
        key: st.session_state[key]
        for key in CALC_STATE_KEYS_TO_PRESERVE
        if key in st.session_state
    }


def _ensure_selected_layer() -> None:
    layers = st.session_state.layers
    valid_ids = [str(layer["id"]) for layer in layers]
    selected = st.session_state.get("selected_layer_id")
    if selected in valid_ids:
        return
    if len(layers) > 2:
        st.session_state.selected_layer_id = str(layers[1]["id"])
    elif layers:
        st.session_state.selected_layer_id = str(layers[0]["id"])
    else:
        st.session_state.selected_layer_id = None


def _layer_role(index: int, total: int) -> str:
    if index == 0:
        return "Superstrate"
    if index == total - 1:
        return "Substrate"
    return "Internal layer"


def _layer_short_label(index: int, total: int, layer: Dict[str, object]) -> str:
    material = str(layer["material"])
    alpha_deg = float(layer.get("alpha", layer.get("alpha_rel_substrate_deg", 0.0)))
    thickness_um = float(layer["thickness_m"]) * 1e6
    role = _layer_role(index, total)
    if index in (0, total - 1):
        return f"{role} • {material} • α {alpha_deg:.1f}°"
    return f"{role} • {material} • {thickness_um:.3f} µm • α {alpha_deg:.1f}°"


def build_stack_from_session() -> StackSpec:
    """Translate the editable Streamlit session state into a validated stack model."""
    layers = []
    total = len(st.session_state.layers)
    for index, layer in enumerate(st.session_state.layers):
        boundary = index in (0, total - 1)
        thickness = 0.0 if boundary else float(layer["thickness_m"])
        if boundary:
            euler = (0.0, 0.0, 0.0)
        else:
            euler = (float(layer["alpha"]), float(layer["beta"]), float(layer["gamma"]))
        layers.append(
            LayerSpec(
                material=str(layer["material"]),
                thickness_m=thickness,
                euler_deg=euler,
                doping=DopingSpec(
                    enabled=bool(layer["doping_enabled"]),
                    wp_cm1=float(layer["wp_cm1"]),
                    gp_cm1=float(layer["gp_cm1"]),
                ),
            )
        )
    return StackSpec.from_layers(layers).enforce_boundary_layers()


def _move_selected_interior_layer(delta: int) -> None:
    layers = st.session_state.layers
    if len(layers) <= 2:
        return
    selected_id = str(st.session_state.get("selected_layer_id"))
    current_index = next((i for i, layer in enumerate(layers) if str(layer["id"]) == selected_id), None)
    if current_index is None or current_index in (0, len(layers) - 1):
        return
    target_index = current_index + int(delta)
    if target_index <= 0 or target_index >= len(layers) - 1:
        return
    moving = layers.pop(current_index)
    layers.insert(target_index, moving)
    st.session_state.layers = layers
    _preserve_calc_state_before_rerun()
    st.rerun()


def _render_layer_row(index: int, total: int, layer: Dict[str, object], selected_id: str) -> None:
    layer_id = str(layer["id"])
    selected = layer_id == selected_id
    boundary = index in (0, total - 1)
    thickness_um = float(layer["thickness_m"]) * 1e6
    alpha_deg = float(layer.get("alpha", layer.get("alpha_rel_substrate_deg", 0.0)))
    role = _layer_role(index, total)
    if index == 0:
        icon = ":material/north:"
        primary_label = f"{icon} Superstrate  |  {layer['material']}"
        secondary_label = "Semi-infinite boundary"
    elif index == total - 1:
        icon = ":material/south:"
        primary_label = f"{icon} Substrate  |  {layer['material']}"
        secondary_label = "Semi-infinite boundary"
    else:
        icon = ":material/layers:"
        primary_label = f"{icon} Layer {index}  |  {layer['material']}"
        secondary_label = f"{thickness_um:.3f} µm  |  α {alpha_deg:.1f}°"
    if boundary:
        secondary_label = f"{secondary_label}  |  α {alpha_deg:.1f}°"
    card_label = f"**{primary_label}**\n{secondary_label}"

    action_cols = st.columns([0.82, 0.06, 0.06, 0.06], gap="small")
    with action_cols[0]:
        if st.button(
            card_label,
            key=layer_widget_key("sel_", layer_id),
            width="stretch",
            type="primary" if selected else "secondary",
        ):
            st.session_state.selected_layer_id = layer_id
            _preserve_calc_state_before_rerun()
            st.rerun()
    if not boundary:
        with action_cols[1]:
            if st.button(":material/arrow_upward:", key=layer_widget_key("move_up_", layer_id), width="stretch"):
                st.session_state.selected_layer_id = layer_id
                _move_selected_interior_layer(-1)
        with action_cols[2]:
            if st.button(":material/arrow_downward:", key=layer_widget_key("move_down_", layer_id), width="stretch"):
                st.session_state.selected_layer_id = layer_id
                _move_selected_interior_layer(1)
        with action_cols[3]:
            if st.button(":material/delete:", key=layer_widget_key("del_", layer_id), width="stretch"):
                st.session_state.layers = [l for l in st.session_state.layers if str(l["id"]) != layer_id]
                _ensure_selected_layer()
                _preserve_calc_state_before_rerun()
                st.rerun()


def _sync_layer_from_widgets(layer: Dict[str, object], layer_id: str, boundary: bool, catalog: Sequence[str]) -> None:
    current_material = str(layer["material"])
    current_thickness_um = float(layer["thickness_m"]) * 1e6
    selector_options = material_selector_options(catalog)
    fallback_material = current_material if current_material in selector_options else selector_options[0]
    material_key = _init_layer_widget_state(layer_id, "mat_", fallback_material)
    st.selectbox(
        "Material",
        selector_options,
        key=material_key,
        on_change=handle_material_selection_change,
        args=material_selection_callback_args(material_key, current_material),
    )
    maybe_open_custom_material_dialog(material_key, catalog)
    layer["material"] = str(st.session_state.get(material_key, current_material))

    if boundary:
        st.caption("Semi-infinite boundary. Thickness and rotation remain fixed.")
        layer["thickness_m"] = 0.0
        layer["alpha_rel_substrate_deg"] = 0.0
        layer["alpha"] = 0.0
        layer["beta"] = 0.0
        layer["gamma"] = 0.0
    else:
        thickness_key = _init_layer_widget_state(layer_id, "thk_", current_thickness_um)
        thickness_um = st.number_input(
            "Thickness (µm)",
            min_value=0.0,
            step=0.05,
            key=thickness_key,
        )
        layer["thickness_m"] = float(thickness_um) * 1e-6

        st.caption("Euler angles (deg)")
        col_a, col_b, col_g = st.columns(3)
        alpha_key = _init_layer_widget_state(layer_id, "a_", float(layer["alpha"]))
        beta_key = _init_layer_widget_state(layer_id, "b_", float(layer["beta"]))
        gamma_key = _init_layer_widget_state(layer_id, "g_", float(layer["gamma"]))
        alpha = col_a.number_input(
            "α",
            step=1.0,
            key=alpha_key,
        )
        beta = col_b.number_input(
            "β",
            step=1.0,
            key=beta_key,
        )
        gamma = col_g.number_input(
            "γ",
            step=1.0,
            key=gamma_key,
        )
        layer["alpha"] = alpha
        layer["alpha_rel_substrate_deg"] = alpha
        layer["beta"] = beta
        layer["gamma"] = gamma

    doping_key = _init_layer_widget_state(layer_id, "dop_", bool(layer["doping_enabled"]))
    with st.expander("Advanced material options", expanded=bool(layer.get("doping_enabled", False)), icon=":material/science:"):
        if boundary:
            st.caption("Drude doping is available for semi-infinite boundary media.")
        dop = st.toggle(
            "Enable Drude doping",
            key=doping_key,
        )
        layer["doping_enabled"] = dop
        if dop:
            wp_key = _init_layer_widget_state(layer_id, "wp_", float(layer["wp_cm1"]))
            gp_key = _init_layer_widget_state(layer_id, "gp_", float(layer["gp_cm1"]))
            col_wp, col_gp = st.columns(2)
            layer["wp_cm1"] = col_wp.number_input(
                "wp (cm⁻¹)",
                min_value=0.0,
                step=10.0,
                key=wp_key,
            )
            layer["gp_cm1"] = col_gp.number_input(
                "Gp (cm⁻¹)",
                min_value=0.0,
                step=1.0,
                key=gp_key,
            )
        else:
            layer["wp_cm1"] = 0.0
            layer["gp_cm1"] = 0.0

    return


def _selected_layer() -> tuple[int, Dict[str, object]]:
    selected_id = str(st.session_state.get("selected_layer_id"))
    selected_idx = next((i for i, layer in enumerate(st.session_state.layers) if str(layer["id"]) == selected_id), 0)
    return selected_idx, st.session_state.layers[selected_idx]


def render_stack_preview() -> None:
    """Render the pseudo-3D stack preview close to the layer editor."""
    st.markdown('<p class="section-label">Stack preview</p>', unsafe_allow_html=True)
    stack_preview = build_stack_from_session()
    figure = plot_stack_pseudo3d(stack_preview)
    st.pyplot(figure, width="stretch", clear_figure=True)
    plt.close(figure)


def render_layer_settings(catalog: Sequence[str], notes: Mapping[str, str]) -> None:
    """Render the editable controls for the currently selected layer."""
    selected_idx, selected_layer = _selected_layer()
    total = len(st.session_state.layers)
    selected_id = str(selected_layer["id"])
    boundary = selected_idx in (0, total - 1)

    st.markdown('<p class="section-label">Selected layer</p>', unsafe_allow_html=True)
    st.caption(_layer_short_label(selected_idx, total, selected_layer))
    _sync_layer_from_widgets(selected_layer, selected_id, boundary, catalog)
    note = notes.get(str(selected_layer["material"])) or custom_material_notes().get(str(selected_layer["material"]))
    if note:
        st.caption(f"{selected_layer['material']}: {note}")


def render_stack_panel(catalog: Sequence[str], notes: Mapping[str, str]) -> None:
    """Render the full stack-builder panel, including presets, ordering, and editing."""
    _cleanup_layer_widget_state(str(layer["id"]) for layer in st.session_state.layers)
    _ensure_selected_layer()

    st.markdown('<p class="section-label">Stack builder</p>', unsafe_allow_html=True)
    st.subheader("Layer sequence", anchor=False)

    # Keep construction actions close to the sequence so stack edits and layer selection read as one workflow.
    with st.container(border=True):
        st.markdown('<p class="section-label">Actions</p>', unsafe_allow_html=True)
        action_cols = st.columns(3, gap="small")
        with action_cols[0]:
            if st.button(":material/add: Add layer", width="stretch"):
                new_layer = _new_layer(material="vac", thickness_m=0.5e-6)
                st.session_state.layers.insert(1, new_layer)
                st.session_state.selected_layer_id = str(new_layer["id"])
                _preserve_calc_state_before_rerun()
                st.rerun()
        with action_cols[1]:
            if st.button(":material/science: Load example stack", width="stretch"):
                st.session_state.layers = example_layer_stack()
                st.session_state.layer_seq = 3
                st.session_state.selected_layer_id = str(st.session_state.layers[1]["id"])
                _preserve_calc_state_before_rerun()
                st.rerun()
        with action_cols[2]:
            if st.button(":material/restart_alt: Reset defaults", width="stretch"):
                st.session_state.layers = default_layer_stack()
                st.session_state.layer_seq = 2
                st.session_state.selected_layer_id = str(st.session_state.layers[0]["id"])
                _preserve_calc_state_before_rerun()
                st.rerun()

        st.markdown('<div class="stack-sequence">', unsafe_allow_html=True)
        total = len(st.session_state.layers)
        selected_id = str(st.session_state.get("selected_layer_id"))
        for index, layer in enumerate(st.session_state.layers):
            _render_layer_row(index, total, layer, selected_id)
        st.markdown("</div>", unsafe_allow_html=True)
        render_layer_settings(catalog=catalog, notes=notes)

    with st.container():
        render_stack_preview()
