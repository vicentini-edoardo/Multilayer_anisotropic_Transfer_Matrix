"""Session-scoped custom material dialog and catalog helpers."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping, Sequence

import streamlit as st

from ..custom_materials import (
    ADD_NEW_MATERIAL_OPTION,
    AXIS_NAMES,
    custom_material_names,
    custom_material_note,
    load_material_definition_json,
    material_definition_to_json_bytes,
    parse_epsilon_table,
)


CUSTOM_MATERIAL_REGISTRY_KEY = "custom_material_registry"
CUSTOM_MATERIAL_DIALOG_TARGET_KEY = "custom_material_dialog_target"
CUSTOM_MATERIAL_DIALOG_PREVIOUS_KEY = "custom_material_dialog_previous"
CUSTOM_MATERIAL_DIALOG_OPEN_KEY = "custom_material_dialog_open"
CUSTOM_MATERIAL_DIALOG_IMPORT_DIGEST_KEY = "custom_material_dialog_import_digest"


def init_custom_material_state() -> None:
    """Initialize the session-scoped custom material registry."""
    st.session_state.setdefault(CUSTOM_MATERIAL_REGISTRY_KEY, {})
    st.session_state.setdefault(CUSTOM_MATERIAL_DIALOG_TARGET_KEY, None)
    st.session_state.setdefault(CUSTOM_MATERIAL_DIALOG_PREVIOUS_KEY, None)
    st.session_state.setdefault(CUSTOM_MATERIAL_DIALOG_OPEN_KEY, False)
    st.session_state.setdefault(CUSTOM_MATERIAL_DIALOG_IMPORT_DIGEST_KEY, None)


def custom_material_registry() -> dict[str, dict[str, Any]]:
    """Return the mutable session-scoped custom material registry."""
    return st.session_state.setdefault(CUSTOM_MATERIAL_REGISTRY_KEY, {})


def material_selector_options(base_catalog: Sequence[str]) -> list[str]:
    """Return built-ins, custom session materials, and the dialog action entry."""
    return list(base_catalog) + custom_material_names(custom_material_registry()) + [ADD_NEW_MATERIAL_OPTION]


def custom_material_notes() -> dict[str, str]:
    """Return user-facing notes for the current session custom materials."""
    return {
        name: custom_material_note(name, custom_material_registry()) or "Custom material"
        for name in custom_material_names(custom_material_registry())
    }


def _dialog_key(name: str) -> str:
    return f"custom_material_dialog_{name}"


def _setdefault(key: str, value: Any) -> None:
    st.session_state.setdefault(_dialog_key(key), value)


def _reset_dialog_keys() -> None:
    for suffix in list(st.session_state.keys()):
        if suffix.startswith("custom_material_dialog_") and suffix not in {
            CUSTOM_MATERIAL_DIALOG_TARGET_KEY,
            CUSTOM_MATERIAL_DIALOG_PREVIOUS_KEY,
            CUSTOM_MATERIAL_DIALOG_OPEN_KEY,
            CUSTOM_MATERIAL_DIALOG_IMPORT_DIGEST_KEY,
        }:
            st.session_state.pop(suffix, None)


def _load_definition_into_dialog(definition: Mapping[str, Any]) -> None:
    _reset_dialog_keys()
    st.session_state[_dialog_key("name")] = str(definition["name"])
    st.session_state[_dialog_key("tensor_mode")] = str(definition["tensor_mode"])
    for axis_name in AXIS_NAMES:
        axis_definition = definition["axes"][axis_name]
        source_type = str(axis_definition["source_type"])
        st.session_state[_dialog_key(f"{axis_name}_source_type")] = {
            "weak_lorentz": "Weak Lorentz",
            "strong_lorentz": "Strong Lorentz",
            "table": "TXT import",
        }[source_type]
        if source_type in {"weak_lorentz", "strong_lorentz"}:
            oscillators = axis_definition["oscillators"]
            st.session_state[_dialog_key(f"{axis_name}_{source_type}_eps_inf")] = float(axis_definition["eps_inf"])
            st.session_state[_dialog_key(f"{axis_name}_{source_type}_count")] = int(len(oscillators))
            for index, oscillator in enumerate(oscillators):
                for field_name, value in oscillator.items():
                    st.session_state[_dialog_key(f"{axis_name}_{source_type}_{field_name}_{index}")] = float(value)
        else:
            st.session_state[_dialog_key(f"{axis_name}_table_data")] = axis_definition["table"]
            st.session_state[_dialog_key(f"{axis_name}_table_filename")] = f"{definition['name']}_{axis_name}.txt"


def _ensure_dialog_defaults() -> None:
    _setdefault("name", "")
    _setdefault("tensor_mode", "Isotropic")
    for axis_name in AXIS_NAMES:
        _setdefault(f"{axis_name}_source_type", "Weak Lorentz")
        _setdefault(f"{axis_name}_weak_lorentz_eps_inf", 1.0)
        _setdefault(f"{axis_name}_strong_lorentz_eps_inf", 1.0)
        _setdefault(f"{axis_name}_weak_lorentz_count", 1)
        _setdefault(f"{axis_name}_strong_lorentz_count", 1)
        _setdefault(f"{axis_name}_table_data", None)
        _setdefault(f"{axis_name}_table_filename", None)
        _setdefault(f"{axis_name}_weak_lorentz_strength_0", 1.0)
        _setdefault(f"{axis_name}_weak_lorentz_w0_cm1_0", 800.0)
        _setdefault(f"{axis_name}_weak_lorentz_gamma_cm1_0", 5.0)
        _setdefault(f"{axis_name}_strong_lorentz_to_cm1_0", 800.0)
        _setdefault(f"{axis_name}_strong_lorentz_lo_cm1_0", 900.0)
        _setdefault(f"{axis_name}_strong_lorentz_gamma_cm1_0", 5.0)


def _material_selection_callback(material_key: str, previous_material: str) -> None:
    if st.session_state.get(material_key) != ADD_NEW_MATERIAL_OPTION:
        return
    st.session_state[CUSTOM_MATERIAL_DIALOG_TARGET_KEY] = material_key
    st.session_state[CUSTOM_MATERIAL_DIALOG_PREVIOUS_KEY] = previous_material
    st.session_state[CUSTOM_MATERIAL_DIALOG_OPEN_KEY] = True
    st.session_state[material_key] = previous_material


def material_selection_callback_args(material_key: str, previous_material: str) -> tuple[str, str]:
    """Return stable callback arguments for the material selector."""
    return material_key, previous_material


def handle_material_selection_change(material_key: str, previous_material: str) -> None:
    """Open the custom material dialog when the special selectbox action is chosen."""
    _material_selection_callback(material_key, previous_material)


def _capture_imported_definition() -> None:
    uploaded = st.file_uploader("Load saved material (.json)", type=["json"], key=_dialog_key("manifest_upload"))
    if uploaded is None:
        return
    digest = sha256(uploaded.getvalue()).hexdigest()
    if st.session_state.get(CUSTOM_MATERIAL_DIALOG_IMPORT_DIGEST_KEY) == digest:
        return
    definition = load_material_definition_json(uploaded.getvalue())
    _load_definition_into_dialog(definition)
    st.session_state[CUSTOM_MATERIAL_DIALOG_IMPORT_DIGEST_KEY] = digest


def _render_axis_definition_editor(axis_name: str) -> None:
    axis_label = f"ε{axis_name}"
    source_key = _dialog_key(f"{axis_name}_source_type")
    source_type = st.selectbox(
        f"{axis_label} source",
        ["Weak Lorentz", "Strong Lorentz", "TXT import"],
        key=source_key,
    )

    if source_type == "Weak Lorentz":
        eps_inf = st.number_input(f"{axis_label} εinf", min_value=0.0, step=0.1, key=_dialog_key(f"{axis_name}_weak_lorentz_eps_inf"))
        count = int(
            st.number_input(
                f"{axis_label} weak oscillators",
                min_value=1,
                max_value=6,
                step=1,
                key=_dialog_key(f"{axis_name}_weak_lorentz_count"),
            )
        )
        for index in range(count):
            cols = st.columns(3, gap="small")
            cols[0].number_input("Strength", min_value=0.0, step=0.1, key=_dialog_key(f"{axis_name}_weak_lorentz_strength_{index}"))
            cols[1].number_input("w0 (cm⁻¹)", min_value=0.0, step=1.0, key=_dialog_key(f"{axis_name}_weak_lorentz_w0_cm1_{index}"))
            cols[2].number_input("γ (cm⁻¹)", min_value=0.0, step=0.1, key=_dialog_key(f"{axis_name}_weak_lorentz_gamma_cm1_{index}"))
        st.caption(f"{axis_label} εinf = {eps_inf:.3f}")
        return

    if source_type == "Strong Lorentz":
        eps_inf = st.number_input(f"{axis_label} εinf", min_value=0.0, step=0.1, key=_dialog_key(f"{axis_name}_strong_lorentz_eps_inf"))
        count = int(
            st.number_input(
                f"{axis_label} strong oscillators",
                min_value=1,
                max_value=6,
                step=1,
                key=_dialog_key(f"{axis_name}_strong_lorentz_count"),
            )
        )
        for index in range(count):
            cols = st.columns(3, gap="small")
            cols[0].number_input("TO (cm⁻¹)", min_value=0.0, step=1.0, key=_dialog_key(f"{axis_name}_strong_lorentz_to_cm1_{index}"))
            cols[1].number_input("LO (cm⁻¹)", min_value=0.0, step=1.0, key=_dialog_key(f"{axis_name}_strong_lorentz_lo_cm1_{index}"))
            cols[2].number_input("γ (cm⁻¹)", min_value=0.0, step=0.1, key=_dialog_key(f"{axis_name}_strong_lorentz_gamma_cm1_{index}"))
        st.caption(f"{axis_label} εinf = {eps_inf:.3f}")
        return

    uploaded = st.file_uploader(
        f"{axis_label} table (.txt)",
        type=["txt", "csv"],
        key=_dialog_key(f"{axis_name}_table_upload"),
        help="Three columns: frequency (cm^-1), Re(eps), Im(eps).",
    )
    if uploaded is not None:
        digest = sha256(uploaded.getvalue()).hexdigest()
        hash_key = _dialog_key(f"{axis_name}_table_digest")
        if st.session_state.get(hash_key) != digest:
            st.session_state[_dialog_key(f"{axis_name}_table_data")] = parse_epsilon_table(uploaded.getvalue())
            st.session_state[_dialog_key(f"{axis_name}_table_filename")] = uploaded.name
            st.session_state[hash_key] = digest
    table_data = st.session_state.get(_dialog_key(f"{axis_name}_table_data"))
    if table_data is not None:
        st.caption(
            f"Loaded {st.session_state.get(_dialog_key(f'{axis_name}_table_filename'), 'table')} "
            f"with {len(table_data['frequency_cm1'])} rows covering "
            f"{table_data['frequency_cm1'][0]:.3f}–{table_data['frequency_cm1'][-1]:.3f} cm⁻¹."
        )


def _build_axis_definition(axis_name: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    source_type = str(st.session_state[_dialog_key(f"{axis_name}_source_type")])
    if source_type == "Weak Lorentz":
        count = int(st.session_state[_dialog_key(f"{axis_name}_weak_lorentz_count")])
        oscillators = []
        for index in range(count):
            strength = float(st.session_state[_dialog_key(f"{axis_name}_weak_lorentz_strength_{index}")])
            w0_cm1 = float(st.session_state[_dialog_key(f"{axis_name}_weak_lorentz_w0_cm1_{index}")])
            gamma_cm1 = float(st.session_state[_dialog_key(f"{axis_name}_weak_lorentz_gamma_cm1_{index}")])
            oscillators.append({"strength": strength, "w0_cm1": w0_cm1, "gamma_cm1": gamma_cm1})
        return {
            "source_type": "weak_lorentz",
            "eps_inf": float(st.session_state[_dialog_key(f"{axis_name}_weak_lorentz_eps_inf")]),
            "oscillators": oscillators,
        }, errors

    if source_type == "Strong Lorentz":
        count = int(st.session_state[_dialog_key(f"{axis_name}_strong_lorentz_count")])
        oscillators = []
        for index in range(count):
            to_cm1 = float(st.session_state[_dialog_key(f"{axis_name}_strong_lorentz_to_cm1_{index}")])
            lo_cm1 = float(st.session_state[_dialog_key(f"{axis_name}_strong_lorentz_lo_cm1_{index}")])
            gamma_cm1 = float(st.session_state[_dialog_key(f"{axis_name}_strong_lorentz_gamma_cm1_{index}")])
            oscillators.append({"to_cm1": to_cm1, "lo_cm1": lo_cm1, "gamma_cm1": gamma_cm1})
        return {
            "source_type": "strong_lorentz",
            "eps_inf": float(st.session_state[_dialog_key(f"{axis_name}_strong_lorentz_eps_inf")]),
            "oscillators": oscillators,
        }, errors

    table_data = st.session_state.get(_dialog_key(f"{axis_name}_table_data"))
    if table_data is None:
        errors.append(f"Upload a TXT table for ε{axis_name}.")
        return None, errors
    return {"source_type": "table", "table": table_data}, errors


def _build_dialog_definition(builtin_catalog: Sequence[str]) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    name = str(st.session_state[_dialog_key("name")]).strip()
    tensor_mode = "isotropic" if st.session_state[_dialog_key("tensor_mode")] == "Isotropic" else "anisotropic_diagonal"

    if not name:
        errors.append("Material name cannot be empty.")
    if name in builtin_catalog:
        errors.append("Material name conflicts with a built-in material.")
    if name in custom_material_registry():
        errors.append("Material name already exists in the current session.")

    axes_to_build = ("xx",) if tensor_mode == "isotropic" else AXIS_NAMES
    built_axes: dict[str, Any] = {}
    for axis_name in axes_to_build:
        axis_definition, axis_errors = _build_axis_definition(axis_name)
        built_axes[axis_name] = axis_definition
        errors.extend(axis_errors)

    if errors:
        return None, errors

    if tensor_mode == "isotropic":
        isotropic_axis = built_axes["xx"]
        axes = {axis_name: isotropic_axis for axis_name in AXIS_NAMES}
    else:
        axes = {axis_name: built_axes[axis_name] for axis_name in AXIS_NAMES}

    return {
        "name": name,
        "tensor_mode": tensor_mode,
        "axes": axes,
    }, errors


@st.dialog("Add custom material", width="large")
def _render_custom_material_dialog(builtin_catalog: Sequence[str]) -> None:
    _ensure_dialog_defaults()
    _capture_imported_definition()

    st.text_input("Material name", key=_dialog_key("name"))
    st.selectbox("Tensor type", ["Isotropic", "Anisotropic diagonal"], key=_dialog_key("tensor_mode"))

    if st.session_state[_dialog_key("tensor_mode")] == "Isotropic":
        with st.container(border=True):
            st.markdown("**Diagonal tensor**")
            st.caption("Isotropic mode reuses the same axis definition for εxx, εyy, and εzz.")
            _render_axis_definition_editor("xx")
    else:
        for axis_name in AXIS_NAMES:
            with st.container(border=True):
                st.markdown(f"**ε{axis_name}**")
                _render_axis_definition_editor(axis_name)

    definition, errors = _build_dialog_definition(builtin_catalog)
    for message in errors:
        st.caption(f":red[• {message}]")

    if definition is not None:
        st.download_button(
            "Download material definition",
            data=material_definition_to_json_bytes(definition),
            file_name=f"{definition['name']}.json",
            mime="application/json",
            width="stretch",
        )

    action_cols = st.columns(2, gap="small")
    if action_cols[0].button("Cancel", width="stretch"):
        target_key = st.session_state.get(CUSTOM_MATERIAL_DIALOG_TARGET_KEY)
        previous_material = st.session_state.get(CUSTOM_MATERIAL_DIALOG_PREVIOUS_KEY)
        if target_key and previous_material is not None:
            st.session_state[target_key] = previous_material
        st.session_state[CUSTOM_MATERIAL_DIALOG_OPEN_KEY] = False
        st.rerun()

    if action_cols[1].button("Add material", width="stretch", type="primary", disabled=definition is None):
        registry = custom_material_registry()
        registry[str(definition["name"])] = definition
        target_key = st.session_state.get(CUSTOM_MATERIAL_DIALOG_TARGET_KEY)
        if target_key:
            st.session_state[target_key] = str(definition["name"])
        st.session_state[CUSTOM_MATERIAL_DIALOG_OPEN_KEY] = False
        st.rerun()


def maybe_open_custom_material_dialog(material_key: str, builtin_catalog: Sequence[str]) -> None:
    """Open the custom material dialog when requested by the material selector."""
    if not bool(st.session_state.get(CUSTOM_MATERIAL_DIALOG_OPEN_KEY, False)):
        return
    if st.session_state.get(CUSTOM_MATERIAL_DIALOG_TARGET_KEY) != material_key:
        return
    _render_custom_material_dialog(builtin_catalog)
