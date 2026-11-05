"""Adapters between the shared built-in catalog and the transfer-matrix solver."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Mapping

import numpy as np
from materials_library import eps_tensor, load, names

from .custom_materials import build_custom_axes
from .models import DopingSpec


CM1_TO_HZ = 299_792_458.0 * 100.0


@dataclass(frozen=True)
class MaterialAxes:
    fx: Callable[[np.ndarray], np.ndarray]
    fy: Callable[[np.ndarray], np.ndarray]
    fz: Callable[[np.ndarray], np.ndarray]
    note: str = ""


def cm1_to_hz(value_cm1: float | np.ndarray) -> np.ndarray:
    return np.asarray(value_cm1) * CM1_TO_HZ


@lru_cache(maxsize=1)
def material_catalog() -> tuple[str, ...]:
    """Built-ins supported by the diagonal-tensor solver."""
    return tuple(
        name for name in names()
        if (definition := load(name))["tensor"] in {"isotropic", "uniaxial", "biaxial"}
        and definition["status"] != "pending"
    )


def material_notes() -> dict[str, str]:
    notes = {}
    for name in material_catalog():
        definition = load(name)
        if definition["status"] != "verified":
            notes[name] = f"{definition['status'].capitalize()} model: {definition['notes'][0]}"
    return notes


def _axis_function(definition: dict, axis: int, scalar_tensor) -> Callable[[np.ndarray], np.ndarray]:
    def evaluate(frequency_hz: np.ndarray) -> np.ndarray:
        frequency = np.asarray(frequency_hz, dtype=float)
        if frequency.ndim == 0:
            return scalar_tensor(float(frequency))[axis, axis]
        values = eps_tensor(definition, frequency.ravel() / CM1_TO_HZ)[:, axis, axis]
        return values.reshape(frequency.shape)

    return evaluate


@lru_cache(maxsize=64)
def _base_axes(material: str) -> MaterialAxes:
    if material not in material_catalog():
        raise KeyError(f"Unknown or unsupported built-in material '{material}'.")
    definition = load(material)

    @lru_cache(maxsize=8)
    def scalar_tensor(frequency_hz: float) -> np.ndarray:
        return eps_tensor(definition, [frequency_hz / CM1_TO_HZ])[0]

    return MaterialAxes(
        fx=_axis_function(definition, 0, scalar_tensor),
        fy=_axis_function(definition, 1, scalar_tensor),
        fz=_axis_function(definition, 2, scalar_tensor),
    )


def _with_doping(
    function: Callable[[np.ndarray], np.ndarray], doping: DopingSpec
) -> Callable[[np.ndarray], np.ndarray]:
    if not doping.enabled or doping.wp_cm1 == 0.0:
        return function

    def evaluate(frequency_hz: np.ndarray) -> np.ndarray:
        w = np.asarray(frequency_hz, dtype=float) / CM1_TO_HZ
        return function(frequency_hz) - doping.wp_cm1**2 / (w**2 + 1j * doping.gp_cm1 * w)

    return evaluate


def axes_for_material(
    material: str,
    doping: DopingSpec | None = None,
    custom_materials: Mapping[str, Mapping[str, Any]] | None = None,
) -> MaterialAxes:
    """Principal-axis responses as functions of frequency in Hz."""
    if custom_materials and material in custom_materials:
        fx, fy, fz, note = build_custom_axes(custom_materials[material])
        base = MaterialAxes(fx, fy, fz, note)
    else:
        base = _base_axes(material)
    doping = doping or DopingSpec()
    return MaterialAxes(
        fx=_with_doping(base.fx, doping),
        fy=_with_doping(base.fy, doping),
        fz=_with_doping(base.fz, doping),
        note=base.note,
    )
