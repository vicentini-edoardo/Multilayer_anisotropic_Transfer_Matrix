"""Validate the in-house transfer-matrix engine against the pyGTM reference.

The numerical engine (``multilayer_atm.solver_fast`` + ``multilayer_atm.engine``)
no longer depends on pyGTM at runtime. pyGTM is kept purely as an independent
implementation to validate against here. The test is skipped if pyGTM is not
installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from multilayer_atm.materials import CM1_TO_HZ, axes_for_material
from multilayer_atm.models import DopingSpec, LayerSpec, StackSpec
from multilayer_atm import solver
from multilayer_atm.solver import passler_to_pygtm_euler

GTM = pytest.importorskip("GTM.GTMcore", reason="pyGTM not installed (validation reference)")


def _pygtm_reference_map(layers, ws_cm1, kx_cm1):
    """Im(rpp) map computed with pyGTM, with its history-dependent layer state
    reset per sample so the result is the clean, order-independent reference."""

    def make(layer):
        axes = axes_for_material(layer.material, layer.doping)
        theta, phi, psi = passler_to_pygtm_euler(layer.euler_deg)
        return GTM.Layer(
            thickness=float(layer.thickness_m),
            epsilon1=axes.fx,
            epsilon2=axes.fy,
            epsilon3=axes.fz,
            theta=theta,
            phi=phi,
            psi=psi,
        )

    system = GTM.System()
    system.set_superstrate(make(layers[0]))
    system.set_substrate(make(layers[-1]))
    for mid in layers[1:-1]:
        system.add_layer(make(mid))

    out = np.empty((len(ws_cm1), len(kx_cm1)))
    for i, w in enumerate(ws_cm1):
        f_hz = float(w * CM1_TO_HZ)
        system.initialize_sys(f_hz)
        for j, kx in enumerate(kx_cm1):
            zeta = complex(kx / w, 1e-14)
            for lay in [system.superstrate, system.substrate, *system.layers]:
                lay._useBerreman = False
                lay.gamma = np.zeros((4, 3), dtype=np.complex128)
            gamma_star = system.calculate_GammaStar(f_hz, zeta)
            denom = gamma_star[0, 0] * gamma_star[2, 2] - gamma_star[0, 2] * gamma_star[2, 0]
            rpp = (
                (gamma_star[1, 0] * gamma_star[2, 2] - gamma_star[1, 2] * gamma_star[2, 0]) / denom
                if denom != 0
                else 0.0
            )
            out[i, j] = float(np.imag(rpp)) if np.isfinite(rpp) else 0.0
    return out


_STACKS = {
    "isotropic": [
        LayerSpec("air", 0.0, (0, 0, 0)),
        LayerSpec("Si", 5e-7, (0, 0, 0)),
        LayerSpec("air", 0.0, (0, 0, 0)),
    ],
    "uniaxial_tilted": [
        LayerSpec("air", 0.0, (0, 0, 0)),
        LayerSpec("hBN", 1e-7, (0, 30, 0)),
        LayerSpec("air", 0.0, (0, 0, 0)),
    ],
    "biaxial": [
        LayerSpec("air", 0.0, (0, 0, 0)),
        LayerSpec("MoO3", 2e-7, (10, 20, 30)),
        LayerSpec("air", 0.0, (0, 0, 0)),
    ],
    "multilayer": [
        LayerSpec("air", 0.0, (0, 0, 0)),
        LayerSpec("hBN", 1e-7, (10, 20, 30)),
        LayerSpec("MoO3", 1.5e-7, (5, 15, 25)),
        LayerSpec("SiC", 1e-6, (0, 0, 0)),
    ],
    "doped_substrate": [
        LayerSpec("air", 0.0, (0, 0, 0)),
        LayerSpec("hBN", 1e-7, (0, 45, 0)),
        LayerSpec("GaAs", 1e-6, (0, 0, 0), DopingSpec(True, 800.0, 50.0)),
    ],
}


@pytest.mark.parametrize("name", list(_STACKS))
@pytest.mark.parametrize("fast", [True, False])
def test_engine_matches_pygtm(name: str, fast: bool) -> None:
    layers = _STACKS[name]
    w_min, w_max, nw = 700.0, 1100.0, 21
    kx_min, kx_max, nk = 0.0, 6.0, 31
    ws = np.linspace(w_min, w_max, nw)
    kx = np.linspace(kx_min, kx_max, nk)

    reference = _pygtm_reference_map(layers, ws, kx)
    _w, _kx, im_rpp = solver.compute_rpp_map(
        StackSpec.from_layers(layers),
        w_min, w_max, nw, kx_min, kx_max, nk,
        workers=1, fast=fast,
    )

    assert np.allclose(im_rpp, reference, rtol=0.0, atol=1e-10)
