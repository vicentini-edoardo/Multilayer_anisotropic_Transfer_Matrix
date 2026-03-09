"""Custom material definitions, serialization, and tensor-axis response builders."""

from __future__ import annotations

from io import StringIO
import json
from typing import Any, Callable, Mapping

import numpy as np


ADD_NEW_MATERIAL_OPTION = "Add new material..."
AXIS_NAMES = ("xx", "yy", "zz")
C_LIGHT = 299_792_458.0
CM1_TO_HZ = C_LIGHT * 100.0


def _cm1_to_hz(value_cm1: float | np.ndarray) -> np.ndarray:
    return np.asarray(value_cm1, dtype=float) * CM1_TO_HZ


def _hz_to_cm1(value_hz: np.ndarray) -> np.ndarray:
    return np.asarray(value_hz, dtype=float) / CM1_TO_HZ


def custom_material_names(registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Return sorted custom material names from the current registry."""
    return sorted(str(name) for name in registry.keys())


def custom_material_note(material_name: str, registry: Mapping[str, Mapping[str, Any]]) -> str | None:
    """Return a short note describing a custom material."""
    definition = registry.get(material_name)
    if definition is None:
        return None
    tensor_mode = str(definition.get("tensor_mode", "anisotropic_diagonal"))
    if tensor_mode == "isotropic":
        return "Custom isotropic material"
    return "Custom anisotropic diagonal material"


def _serialize_axis(axis_definition: Mapping[str, Any]) -> dict[str, Any]:
    source_type = str(axis_definition["source_type"])
    payload: dict[str, Any] = {"source_type": source_type}
    if source_type in {"weak_lorentz", "strong_lorentz"}:
        payload["eps_inf"] = float(axis_definition["eps_inf"])
        payload["oscillators"] = [{k: float(v) for k, v in osc.items()} for osc in axis_definition["oscillators"]]
    elif source_type == "table":
        payload["table"] = {
            "frequency_cm1": [float(v) for v in axis_definition["table"]["frequency_cm1"]],
            "eps_real": [float(v) for v in axis_definition["table"]["eps_real"]],
            "eps_imag": [float(v) for v in axis_definition["table"]["eps_imag"]],
        }
    else:
        raise ValueError(f"Unsupported axis source type '{source_type}'.")
    return payload


def normalize_material_definition(definition: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a loaded material definition into the internal registry format."""
    name = str(definition["name"]).strip()
    if not name:
        raise ValueError("Material name cannot be empty.")

    tensor_mode = str(definition.get("tensor_mode", "anisotropic_diagonal"))
    axes_input = definition.get("axes")
    if not isinstance(axes_input, Mapping):
        raise ValueError("Material definition must provide an 'axes' mapping.")

    axes: dict[str, dict[str, Any]] = {}
    for axis_name in AXIS_NAMES:
        axis_definition = axes_input.get(axis_name)
        if axis_definition is None:
            raise ValueError(f"Material definition is missing axis '{axis_name}'.")
        axes[axis_name] = _serialize_axis(axis_definition)

    return {
        "name": name,
        "tensor_mode": tensor_mode,
        "axes": axes,
    }


def material_definition_to_json_bytes(definition: Mapping[str, Any]) -> bytes:
    """Serialize a custom material definition into the JSON interchange format."""
    normalized = normalize_material_definition(definition)
    return json.dumps(normalized, indent=2, sort_keys=True).encode("utf-8")


def load_material_definition_json(payload: bytes | str) -> dict[str, Any]:
    """Load a custom material definition from the JSON interchange format."""
    raw = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else str(payload)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("The uploaded JSON material file is not valid.") from exc
    if not isinstance(data, Mapping):
        raise ValueError("The uploaded JSON material file must contain an object at the top level.")
    return normalize_material_definition(data)


def parse_epsilon_table(file_bytes: bytes) -> dict[str, list[float]]:
    """Parse a frequency/Re(eps)/Im(eps) TXT table in cm^-1."""
    parse_errors: list[Exception] = []
    for delimiter in (None, ","):
        try:
            table = np.genfromtxt(StringIO(file_bytes.decode("utf-8")), comments="#", delimiter=delimiter, dtype=float)
            if table.ndim == 1:
                table = np.atleast_2d(table)
            if table.shape[1] != 3:
                raise ValueError("Expected exactly three columns: frequency, Re(eps), Im(eps).")
            if np.isnan(table).any():
                raise ValueError("Table contains invalid numeric values.")
            freq = np.asarray(table[:, 0], dtype=float)
            if freq.size < 2:
                raise ValueError("Table must contain at least two frequency rows.")
            if not np.all(np.diff(freq) > 0):
                raise ValueError("Frequency values must be strictly increasing.")
            return {
                "frequency_cm1": freq.tolist(),
                "eps_real": np.asarray(table[:, 1], dtype=float).tolist(),
                "eps_imag": np.asarray(table[:, 2], dtype=float).tolist(),
            }
        except Exception as exc:  # pragma: no cover - exercised through UI uploads
            parse_errors.append(exc)
    raise ValueError(str(parse_errors[-1]) if parse_errors else "Could not parse epsilon table.")


def _weak_lorentz_response(axis_definition: Mapping[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    eps_inf = float(axis_definition["eps_inf"])
    oscillators = axis_definition["oscillators"]

    def _eps(freq_hz: np.ndarray) -> np.ndarray:
        freq_cm1 = _hz_to_cm1(np.asarray(freq_hz, dtype=float))
        response = np.full(freq_cm1.shape, eps_inf + 0.0j, dtype=np.complex128)
        for oscillator in oscillators:
            strength = float(oscillator["strength"])
            w0 = float(oscillator["w0_cm1"])
            gamma = float(oscillator["gamma_cm1"])
            denom = (w0**2 - freq_cm1**2) - 1j * gamma * freq_cm1
            response += strength * (w0**2) / denom
        return response

    return _eps


def _strong_lorentz_response(axis_definition: Mapping[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    eps_inf = float(axis_definition["eps_inf"])
    oscillators = axis_definition["oscillators"]

    def _eps(freq_hz: np.ndarray) -> np.ndarray:
        freq_cm1 = _hz_to_cm1(np.asarray(freq_hz, dtype=float))
        response = np.full(freq_cm1.shape, eps_inf + 0.0j, dtype=np.complex128)
        for oscillator in oscillators:
            to_cm1 = float(oscillator["to_cm1"])
            lo_cm1 = float(oscillator["lo_cm1"])
            gamma = float(oscillator["gamma_cm1"])
            numerator = (lo_cm1**2 - freq_cm1**2) - 1j * gamma * freq_cm1
            denominator = (to_cm1**2 - freq_cm1**2) - 1j * gamma * freq_cm1
            response *= numerator / denominator
        return response

    return _eps


def _table_response(axis_definition: Mapping[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    table = axis_definition["table"]
    frequencies = np.asarray(table["frequency_cm1"], dtype=float)
    eps_real = np.asarray(table["eps_real"], dtype=float)
    eps_imag = np.asarray(table["eps_imag"], dtype=float)

    def _eps(freq_hz: np.ndarray) -> np.ndarray:
        freq_cm1 = _hz_to_cm1(np.asarray(freq_hz, dtype=float))
        clipped = np.clip(freq_cm1, frequencies[0], frequencies[-1])
        real_part = np.interp(clipped, frequencies, eps_real)
        imag_part = np.interp(clipped, frequencies, eps_imag)
        return np.asarray(real_part + 1j * imag_part, dtype=np.complex128)

    return _eps


def build_axis_response(axis_definition: Mapping[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    """Build an epsilon(freq) callable for one tensor axis."""
    source_type = str(axis_definition["source_type"])
    if source_type == "weak_lorentz":
        return _weak_lorentz_response(axis_definition)
    if source_type == "strong_lorentz":
        return _strong_lorentz_response(axis_definition)
    if source_type == "table":
        return _table_response(axis_definition)
    raise KeyError(f"Unsupported custom material axis source '{source_type}'.")


def build_custom_axes(material_definition: Mapping[str, Any]) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], str]:
    """Build principal-axis callables for a full custom material definition."""
    axes = material_definition["axes"]
    note = custom_material_note(str(material_definition["name"]), {str(material_definition["name"]): material_definition}) or "Custom material"
    return (
        build_axis_response(axes["xx"]),
        build_axis_response(axes["yy"]),
        build_axis_response(axes["zz"]),
        note,
    )


def material_frequency_range_warnings(material_names: list[str], registry: Mapping[str, Mapping[str, Any]], w_min_cm1: float, w_max_cm1: float) -> list[str]:
    """Return warnings for custom table-backed materials used outside their table range."""
    warnings: list[str] = []
    for material_name in dict.fromkeys(material_names):
        definition = registry.get(material_name)
        if definition is None:
            continue
        for axis_name in AXIS_NAMES:
            axis_definition = definition["axes"][axis_name]
            if axis_definition["source_type"] != "table":
                continue
            frequencies = np.asarray(axis_definition["table"]["frequency_cm1"], dtype=float)
            if float(w_min_cm1) < float(frequencies[0]) or float(w_max_cm1) > float(frequencies[-1]):
                warnings.append(
                    f"{material_name} ε{axis_name} table covers {frequencies[0]:.3f}–{frequencies[-1]:.3f} cm⁻¹; values outside this range are clamped to the nearest endpoint."
                )
    return warnings
