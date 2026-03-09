"""Built-in presets for the public application."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List


SPEED_PRESETS: dict[str, dict[str, dict[str, int]]] = {
    "Coarse": {
        "map": {"nw": 80, "nk": 100},
        "iso": {"nk": 64, "nphi": 64},
    },
    "Normal": {
        "map": {"nw": 160, "nk": 180},
        "iso": {"nk": 128, "nphi": 128},
    },
    "Fine": {
        "map": {"nw": 260, "nk": 280},
        "iso": {"nk": 256, "nphi": 128},
    },
}


def _layer_record(layer_id: str, material: str, thickness_m: float) -> dict[str, object]:
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


DEFAULT_LAYER_STACK: List[Dict[str, object]] = [
    _layer_record("layer_0", "vac", 0.0),
    _layer_record("layer_1", "Si", 0.0),
]


EXAMPLE_LAYER_STACK: List[Dict[str, object]] = [
    _layer_record("layer_0", "vac", 0.0),
    _layer_record("layer_2", "SiC", 0.1e-6),
    _layer_record("layer_1", "Si", 0.0),
]


def default_layer_stack() -> List[Dict[str, object]]:
    """Return a fresh copy of the default semi-infinite boundary stack."""
    return deepcopy(DEFAULT_LAYER_STACK)


def example_layer_stack() -> List[Dict[str, object]]:
    """Return a minimal example stack that produces a useful public demo."""
    return deepcopy(EXAMPLE_LAYER_STACK)
