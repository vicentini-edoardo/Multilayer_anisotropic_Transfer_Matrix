"""Physical and integration checks for the vertical-dipole wave map."""

import numpy as np
import pytest

from multilayer_atm import solver
from multilayer_atm.models import LayerSpec, StackSpec


def test_isofrequency_retains_phase_and_fallback_rotation(monkeypatch):
    stack = StackSpec.from_layers([
        LayerSpec("air", 0.0), LayerSpec("hBN", 100e-9, (10.0, 30.0, 20.0)),
        LayerSpec("air", 0.0),
    ])
    params = dict(w0=800.0, kx_min=100.0, kx_max=5000.0, nk=12, nphi=8, fast=True)
    phi, k, im = solver.compute_isofreq_map(stack, workers=1, **params)
    _, _, rp = solver.compute_isofreq_map(stack, workers=1, return_complex=True, **params)
    assert np.iscomplexobj(rp)
    assert np.any(np.abs(rp.real) > 1e-6)
    np.testing.assert_allclose(rp.imag, im)

    class UnavailablePool:
        def __init__(self, **kwargs):
            pass

        def map(self, *args, **kwargs):
            raise OSError("No multiprocessing")

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(solver, "ProcessPoolExecutor", UnavailablePool)
    _, _, fallback = solver.compute_isofreq_map(stack, workers=2, return_complex=True, **params)
    np.testing.assert_allclose(fallback, rp, rtol=1e-10, atol=1e-10)


def test_vertical_dipole_matches_electrostatic_image_and_retains_phase():
    from multilayer_atm.surface_wave import compute_surface_wave

    phi = np.linspace(0.0, 2 * np.pi, 128, endpoint=False)
    k = np.linspace(0.0, 500000.0, 600)
    rp = np.ones((len(phi), len(k)), dtype=complex)
    args = dict(w0_cm1=0.001, source_height_nm=25.0, observation_height_nm=75.0, fft_size=1024)
    xy, field, intensity = compute_surface_wave(phi, k, rp, **args)
    c = len(xy) // 2
    # Electrostatic image of a z dipole: Ez ∝ (2h²-r²)/(h²+r²)^(5/2).
    h = 0.1  # um, sum of source and observation heights
    expected = ((2 * h**2 - xy**2) / (h**2 + xy**2)**2.5 / (2 / h**3))**2
    inner = np.abs(xy) <= 0.5
    np.testing.assert_allclose(intensity[c, inner], expected[inner], atol=0.003)
    np.testing.assert_allclose(intensity[c, :], intensity[:, c], atol=1e-12)
    assert np.iscomplexobj(field)
    assert intensity[c, c] == pytest.approx(1.0)

    # A phase ramp translates the launched field; Im(rp) alone cannot do this.
    shift = 0.1  # um, 40 pixels on this FFT grid
    translated_rp = rp * np.exp(-2j * np.pi * (k[None, :] * 1e-4) * np.cos(phi[:, None]) * shift)
    _, _, translated = compute_surface_wave(phi, k, translated_rp, **args)
    assert xy[np.argmax(translated[c])] == pytest.approx(shift)

    _, _, zero = compute_surface_wave(phi, k, rp * 0, **args)
    assert np.count_nonzero(zero) == 0
    with pytest.raises(ValueError, match="360"):
        compute_surface_wave(phi[:64], k, rp[:64], **args)
    with pytest.raises(ValueError, match="complex"):
        compute_surface_wave(phi, k, rp.real, **args)
    with pytest.raises(ValueError, match="height"):
        compute_surface_wave(phi, k, rp, **{**args, "source_height_nm": -1.0})


def test_gui_adds_wave_plot_and_preserves_saved_frequency():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string('''
import numpy as np
import streamlit as st
from multilayer_atm.presets import SPEED_PRESETS
from multilayer_atm.ui.layer_builder import init_layer_state
from multilayer_atm.ui import calculation_views as cv
init_layer_state()
cv.init_calculation_state(SPEED_PRESETS)
if not st.session_state.iso_history:
    phi = np.linspace(0, 2*np.pi, 64, endpoint=False)
    k = np.linspace(500, 25000, 128)
    cv._append_iso_history(phi, k, np.ones((64,128), dtype=complex)*(1+1j))
    st.session_state.iso_state['w0'] = 1500.0
cv._sync_selected_history_result('iso')
cv._render_iso_plot('Normal', 1)
''', default_timeout=30).run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 2
    assert any("900.0 cm" in item.value for item in app.caption)
    wave_spec = app.get("plotly_chart")[1].proto.spec
    app.number_input(key="wave_source_height_nm").set_value(75.0).run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 2
    assert app.get("plotly_chart")[1].proto.spec != wave_spec
