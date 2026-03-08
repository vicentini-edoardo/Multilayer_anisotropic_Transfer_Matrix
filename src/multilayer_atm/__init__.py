"""Public package exports for the multilayer anisotropic transfer-matrix app."""

from .materials import axes_for_material, material_catalog, material_notes
from .models import DopingSpec, LayerSpec, StackSpec
from .solver import compute_isofreq_map, compute_rpp_map

__all__ = [
    "DopingSpec",
    "LayerSpec",
    "StackSpec",
    "axes_for_material",
    "material_catalog",
    "material_notes",
    "compute_rpp_map",
    "compute_isofreq_map",
]
