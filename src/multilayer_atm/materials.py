"""Material models and utility functions for the anisotropic transfer-matrix solver."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Mapping, Tuple

import numpy as np

try:
    import GTM.GTMcore as GTM  # type: ignore
    import GTM.Permittivities as mat  # type: ignore
except Exception as exc:  # pragma: no cover - exercised by runtime environment setup
    raise ImportError(
        "pyGTM is required but not importable. Install a compatible pyGTM package "
        "(for example via `pip install \"pyGTM @ git+https://github.com/pyMatJ/pyGTM.git@7a228b7314ea66ae025ff346c0d0e8bfb86cc82c\"`) and retry."
    ) from exc

from .models import DopingSpec
from .custom_materials import build_custom_axes


C_LIGHT = 299_792_458.0
CM1_TO_HZ = C_LIGHT * 100.0

PASSLER_MATERIALS: Tuple[str, ...] = (
    "vac",
    "air",
    "KRS5",
    "Au",
    "Ag",
    "Si",
    "Ge",
    "Diamond",
    "MgO",
    "SiC4H",
    "SiC6H",
    "SiC",
    "AlN",
    "GaN",
    "InN",
    "hBN",
    "SiO2",
    "Al2O3",
    "BaTiO3",
    "MoO3",
    "GaAs",
    "GaP",
    "InAs",
    "BaF2",
)


@dataclass(frozen=True)
class MaterialAxes:
    """Permittivity callables for the principal material axes."""

    fx: Callable[[np.ndarray], np.ndarray]
    fy: Callable[[np.ndarray], np.ndarray]
    fz: Callable[[np.ndarray], np.ndarray]
    note: str = ""


def cm1_to_hz(value_cm1: float | np.ndarray) -> float | np.ndarray:
    """Convert wavenumbers in cm⁻¹ to frequencies in Hz."""
    return np.asarray(value_cm1) * CM1_TO_HZ


def _const_eps(value: float) -> Callable[[np.ndarray], np.ndarray]:
    def _eps(f_hz: np.ndarray) -> np.ndarray:
        f = np.asarray(f_hz)
        return (value + 0.0j) * np.ones_like(f, dtype=np.complex128)

    return _eps


def _eps_sic4h_x(f_hz: np.ndarray) -> np.ndarray:
    return mat.eps_1phonon(
        np.asarray(f_hz),
        797.0 * CM1_TO_HZ,
        970.0 * CM1_TO_HZ,
        3.75 * CM1_TO_HZ,
        3.75 * CM1_TO_HZ,
        6.5,
    )


def _eps_sic4h_z(f_hz: np.ndarray) -> np.ndarray:
    return mat.eps_1phonon(
        np.asarray(f_hz),
        788.0 * CM1_TO_HZ,
        964.0 * CM1_TO_HZ,
        3.75 * CM1_TO_HZ,
        3.75 * CM1_TO_HZ,
        6.5,
    )


@lru_cache(maxsize=64)
def _base_axes(material: str) -> MaterialAxes:
    m = material
    if m in ("vac", "air"):
        return MaterialAxes(GTM.vacuum_eps, GTM.vacuum_eps, GTM.vacuum_eps)
    if m == "KRS5":
        return MaterialAxes(mat.eps_KRS5, mat.eps_KRS5, mat.eps_KRS5)
    if m == "Au":
        return MaterialAxes(mat.eps_Au, mat.eps_Au, mat.eps_Au)
    if m == "Ag":
        return MaterialAxes(mat.eps_Ag, mat.eps_Ag, mat.eps_Ag)
    if m == "Si":
        c13 = _const_eps(13.0)
        return MaterialAxes(c13, c13, c13)
    if m == "Ge":
        c16 = _const_eps(16.0)
        return MaterialAxes(c16, c16, c16)
    if m == "Diamond":
        cdi = _const_eps(5.6539)
        return MaterialAxes(cdi, cdi, cdi)
    if m == "MgO":
        # Passler source references MgO in the switch list but does not define epsT_MgO.
        # We keep an explicit fallback so the catalog remains complete.
        cmgo = _const_eps(2.95)
        return MaterialAxes(cmgo, cmgo, cmgo, note="Fallback constant epsilon")
    if m == "SiC4H":
        return MaterialAxes(_eps_sic4h_x, _eps_sic4h_x, _eps_sic4h_z)
    if m == "SiC6H":
        return MaterialAxes(mat.eps_SiC6Hx, mat.eps_SiC6Hx, mat.eps_SiC6Hz)
    if m == "SiC":
        return MaterialAxes(mat.eps_SiCx, mat.eps_SiCx, mat.eps_SiCz)
    if m == "AlN":
        return MaterialAxes(mat.eps_AlNx, mat.eps_AlNx, mat.eps_AlNz)
    if m == "GaN":
        return MaterialAxes(mat.eps_GaNx, mat.eps_GaNx, mat.eps_GaNz)
    if m == "InN":
        return MaterialAxes(mat.eps_InNx, mat.eps_InNx, mat.eps_InNz)
    if m == "hBN":
        return MaterialAxes(mat.eps_hBNx, mat.eps_hBNx, mat.eps_hBNz)
    if m == "SiO2":
        # pyGTM ships isotropic SiO2; MATLAB code uses anisotropic alpha-quartz tables.
        return MaterialAxes(mat.eps_SiO2, mat.eps_SiO2, mat.eps_SiO2, note="Isotropic pyGTM model")
    if m == "Al2O3":
        return MaterialAxes(mat.eps_Al2O3o, mat.eps_Al2O3o, mat.eps_Al2O3e)
    if m == "BaTiO3":
        return MaterialAxes(mat.eps_BaTiO3, mat.eps_BaTiO3, mat.eps_BaTiO3)
    if m == "MoO3":
        return MaterialAxes(mat.eps_MoO3x, mat.eps_MoO3y, mat.eps_MoO3z)
    if m == "GaAs":
        return MaterialAxes(mat.eps_GaAs, mat.eps_GaAs, mat.eps_GaAs)
    if m == "GaP":
        return MaterialAxes(mat.eps_GaP, mat.eps_GaP, mat.eps_GaP)
    if m == "InAs":
        return MaterialAxes(mat.eps_InAs, mat.eps_InAs, mat.eps_InAs)
    if m == "BaF2":
        return MaterialAxes(mat.eps_BaF2, mat.eps_BaF2, mat.eps_BaF2)
    raise KeyError(f"Unknown material '{material}'.")


def _with_doping(
    fn: Callable[[np.ndarray], np.ndarray],
    doping: DopingSpec,
) -> Callable[[np.ndarray], np.ndarray]:
    if not doping.enabled:
        return fn
    if doping.wp_cm1 == 0.0:
        return fn

    wp_hz = float(cm1_to_hz(doping.wp_cm1))
    gp_hz = float(cm1_to_hz(doping.gp_cm1))

    def _eps(f_hz: np.ndarray) -> np.ndarray:
        f = np.asarray(f_hz)
        return fn(f) + mat.eps_drude(f, wp_hz, gp_hz)

    return _eps


def axes_for_material(
    material: str,
    doping: DopingSpec | None = None,
    custom_materials: Mapping[str, Mapping[str, Any]] | None = None,
) -> MaterialAxes:
    """Return the anisotropic permittivity model for a supported material."""
    if custom_materials and material in custom_materials:
        fx, fy, fz, note = build_custom_axes(custom_materials[material])
        base = MaterialAxes(fx=fx, fy=fy, fz=fz, note=note)
    else:
        base = _base_axes(material)
    doping = doping or DopingSpec()
    return MaterialAxes(
        fx=_with_doping(base.fx, doping),
        fy=_with_doping(base.fy, doping),
        fz=_with_doping(base.fz, doping),
        note=base.note,
    )


def material_catalog() -> Tuple[str, ...]:
    """Return the supported material names exposed by the public app."""
    return PASSLER_MATERIALS


def material_notes() -> Dict[str, str]:
    """Return human-readable notes for materials with fallback or simplified models."""
    notes: Dict[str, str] = {}
    for m in PASSLER_MATERIALS:
        note = _base_axes(m).note
        if note:
            notes[m] = note
    return notes
