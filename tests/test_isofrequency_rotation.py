from __future__ import annotations

import numpy as np

from multilayer_atm.models import DopingSpec, LayerSpec, StackSpec
from multilayer_atm.solver import compute_isofreq_map


def _three_layer_stack(euler_deg: tuple[float, float, float]) -> StackSpec:
    return StackSpec.from_layers(
        [
            LayerSpec("air", 0.0, (0.0, 0.0, 0.0), DopingSpec()),
            LayerSpec("hBN", 100e-9, euler_deg, DopingSpec()),
            LayerSpec("air", 0.0, (0.0, 0.0, 0.0), DopingSpec()),
        ]
    )


def test_isofrequency_sweep_varies_for_tilted_uniaxial_layer() -> None:
    stack = _three_layer_stack((0.0, 30.0, 0.0))

    _phi_values, _kx_values, im_rpp = compute_isofreq_map(
        stack,
        w0=800.0,
        kx_min=100.0,
        kx_max=500.0,
        nk=5,
        nphi=8,
        workers=1,
    )

    row_spread = np.max(np.abs(im_rpp - im_rpp[0:1, :]), axis=1)

    assert np.any(row_spread[1:] > 1e-8)
