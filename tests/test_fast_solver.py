"""The vectorised ("fast") solver must match the per-point solver numerically."""

from __future__ import annotations

import numpy as np

from multilayer_atm.models import DopingSpec, LayerSpec, StackSpec
from multilayer_atm.solver import compute_isofreq_map, compute_rpp_map


def _biaxial_stack() -> StackSpec:
    return StackSpec.from_layers(
        [
            LayerSpec("air", 0.0, (0.0, 0.0, 0.0), DopingSpec()),
            LayerSpec("hBN", 100e-9, (10.0, 20.0, 30.0), DopingSpec()),
            LayerSpec("MoO3", 150e-9, (5.0, 15.0, 25.0), DopingSpec()),
            LayerSpec("SiC", 1e-6, (0.0, 0.0, 0.0), DopingSpec()),
        ]
    )


def test_fast_dispersion_matches_per_point() -> None:
    stack = _biaxial_stack()
    common = dict(w_min=700.0, w_max=1000.0, nw=18, kx_min=0.0, kx_max=500.0, nk=24, workers=1)
    _w, _kx, slow = compute_rpp_map(stack, fast=False, **common)
    _w2, _kx2, fast = compute_rpp_map(stack, fast=True, **common)
    assert np.max(np.abs(slow - fast)) < 1e-10


def test_fast_isofrequency_matches_per_point() -> None:
    stack = _biaxial_stack()
    common = dict(w0=850.0, kx_min=0.0, kx_max=500.0, nk=20, nphi=16, workers=1)
    _p, _kx, slow = compute_isofreq_map(stack, fast=False, **common)
    _p2, _kx2, fast = compute_isofreq_map(stack, fast=True, **common)
    assert np.max(np.abs(slow - fast)) < 1e-10
