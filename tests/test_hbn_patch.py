from __future__ import annotations

import numpy as np
import pytest

import GTM.Permittivities as mat

from multilayer_atm.materials import CM1_TO_HZ, axes_for_material


def test_hbn_uses_local_z_axis_model_when_upstream_hbnz_is_broken(monkeypatch: pytest.MonkeyPatch) -> None:
    def _broken_eps_hbnz(_f: np.ndarray) -> np.ndarray:
        raise RuntimeError("broken upstream hBNz should not be called")

    monkeypatch.setattr(mat, "eps_hBNz", _broken_eps_hbnz)

    axes = axes_for_material("hBN")
    freq_hz = np.array([780.0 * CM1_TO_HZ], dtype=float)

    eps_x = axes.fx(freq_hz)
    eps_z = axes.fz(freq_hz)

    assert np.isfinite(eps_x).all()
    assert np.isfinite(eps_z).all()
