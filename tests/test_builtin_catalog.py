import numpy as np

from materials_library import eps_tensor, load
from multilayer_atm.materials import CM1_TO_HZ, axes_for_material, material_catalog
from multilayer_atm.models import DopingSpec


def test_builtin_catalog_filters_unsupported_models():
    catalog = material_catalog()
    assert "vacuum" in catalog and "SiC3C" in catalog and "aSiO2" in catalog
    assert not {"graphene", "Ga2O3_beta", "BaTiO3", "vac", "SiC"} & set(catalog)


def test_builtin_axes_match_shared_library_with_doping():
    frequency_cm1 = np.array([800.0, 1000.0])
    axes = axes_for_material("hBN")
    expected = eps_tensor(load("hBN"), frequency_cm1)
    np.testing.assert_allclose(axes.fx(frequency_cm1 * CM1_TO_HZ), expected[:, 0, 0])
    np.testing.assert_allclose(axes.fz(frequency_cm1 * CM1_TO_HZ), expected[:, 2, 2])

    doped = axes_for_material("hBN", DopingSpec(True, 100.0, 10.0))
    correction = -100.0**2 / (frequency_cm1**2 + 1j * 10.0 * frequency_cm1)
    np.testing.assert_allclose(doped.fx(frequency_cm1 * CM1_TO_HZ), expected[:, 0, 0] + correction)


def test_scalar_axis_calls_evaluate_tensor_once(monkeypatch):
    import multilayer_atm.materials as materials

    materials._base_axes.cache_clear()
    original = materials.eps_tensor
    calls = []

    def counting_tensor(*args):
        calls.append(args[1])
        return original(*args)

    monkeypatch.setattr(materials, "eps_tensor", counting_tensor)
    axes = materials.axes_for_material("hBN")
    frequency = 800.0 * CM1_TO_HZ
    axes.fx(frequency)
    axes.fy(frequency)
    axes.fz(frequency)
    assert len(calls) == 1
