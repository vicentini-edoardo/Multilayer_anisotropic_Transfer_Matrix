"""Immutable stack data models used by the solver and Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class DopingSpec:
    """Drude-term parameters applied to a material model."""

    enabled: bool = False
    wp_cm1: float = 0.0
    gp_cm1: float = 0.0


@dataclass(frozen=True)
class LayerSpec:
    """A single layer in the optical stack."""

    material: str
    thickness_m: float
    euler_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # alpha, beta, gamma
    doping: DopingSpec = field(default_factory=DopingSpec)


@dataclass(frozen=True)
class StackSpec:
    """An ordered multilayer stack from superstrate to substrate."""

    layers: Tuple[LayerSpec, ...]

    @classmethod
    def from_layers(cls, layers: Iterable[LayerSpec]) -> "StackSpec":
        """Build a frozen stack from any iterable of layers."""
        return cls(layers=tuple(layers))

    def enforce_boundary_layers(self) -> "StackSpec":
        """Force the first and last layers to remain semi-infinite boundaries."""
        if len(self.layers) < 2:
            raise ValueError("A stack must contain at least two layers (superstrate and substrate).")

        updated: List[LayerSpec] = list(self.layers)
        first = updated[0]
        last = updated[-1]
        updated[0] = LayerSpec(
            material=first.material,
            thickness_m=0.0,
            euler_deg=first.euler_deg,
            doping=first.doping,
        )
        updated[-1] = LayerSpec(
            material=last.material,
            thickness_m=0.0,
            euler_deg=last.euler_deg,
            doping=last.doping,
        )
        return StackSpec.from_layers(updated)

    def with_interior_alpha_offset(self, phi_deg: float) -> "StackSpec":
        """Apply a common in-plane alpha offset to all layers."""
        shifted: List[LayerSpec] = []
        for layer in self.layers:
            alpha, beta, gamma = layer.euler_deg
            shifted.append(
                LayerSpec(
                    material=layer.material,
                    thickness_m=layer.thickness_m,
                    euler_deg=(alpha + phi_deg, beta, gamma),
                    doping=layer.doping,
                )
            )
        return StackSpec.from_layers(shifted)

    def with_global_phi_offset(self, phi_deg: float) -> "StackSpec":
        """Compatibility alias for older call sites that used phi-offset naming."""
        return self.with_interior_alpha_offset(phi_deg)

    def validate(self) -> None:
        """Validate basic geometric and Euler-angle invariants for the stack."""
        if len(self.layers) < 2:
            raise ValueError("A stack must contain at least two layers.")
        for i, layer in enumerate(self.layers):
            if layer.thickness_m < 0:
                raise ValueError(f"Layer {i} has negative thickness.")
            if len(layer.euler_deg) != 3:
                raise ValueError(f"Layer {i} must provide three Euler angles.")
