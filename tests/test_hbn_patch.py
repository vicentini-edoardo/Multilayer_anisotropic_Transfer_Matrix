"""The hBN model comes from the shared catalog."""

import numpy as np

from materials_library import eps_tensor, load
from multilayer_atm.materials import CM1_TO_HZ, axes_for_material


def test_hbn_uses_shared_catalog_model() -> None:
    frequency_cm1 = np.array([780.0])
    axes = axes_for_material("hBN")
    tensor = eps_tensor(load("hBN"), frequency_cm1)
    np.testing.assert_allclose(axes.fx(frequency_cm1 * CM1_TO_HZ), tensor[:, 0, 0])
    np.testing.assert_allclose(axes.fz(frequency_cm1 * CM1_TO_HZ), tensor[:, 2, 2])
