"""Physical and integration checks for the vertical-dipole wave map."""

import numpy as np
import pytest

from multilayer_atm import solver
from multilayer_atm.models import LayerSpec, StackSpec


def test_isofrequency_retains_phase_and_fallback_rotation(monkeypatch):
    stack = StackSpec.from_layers([
        LayerSpec("vacuum", 0.0), LayerSpec("hBN", 100e-9, (10.0, 30.0, 20.0)),
        LayerSpec("vacuum", 0.0),
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
cv.render_wave_options()
cv._render_surface_wave_plot()
''', default_timeout=30).run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 2
    assert app.session_state.iso_history[0]["w0_cm1"] == 900.0
    wave_spec = app.get("plotly_chart")[1].proto.spec
    app.number_input(key="wave_source_height_nm").set_value(75.0).run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 2
    assert app.get("plotly_chart")[1].proto.spec != wave_spec


def test_anisotropic_field_matches_direct_polar_quadrature():
    """FFT + polar interpolation against a plain sum over (q, phi), off-axis points, phi-dependent complex rpp."""
    from multilayer_atm.surface_wave import compute_surface_wave

    phi = np.linspace(0.0, 2 * np.pi, 360, endpoint=False)
    k = np.linspace(100.0, 1.0e5, 500)
    w0, h = 900.0, 25.0
    q0 = 3.0e4 * (1 + 0.35 * np.cos(2 * phi))
    rp = 1 + 0.5 / (1 - (k[None, :] / q0[:, None]) ** 2 + 0.03j)
    xy, field, _ = compute_surface_wave(phi, k, rp, w0_cm1=w0, source_height_nm=h, fft_size=2048)

    q = k * 1e-4
    kappa = 2 * np.pi * np.sqrt(np.maximum(q**2 - (w0 * 1e-4) ** 2, 1e-30))
    spec = -1j * (2 * np.pi * q) ** 2 / kappa * np.exp(-kappa * h * 1e-3) * rp
    spec[:, q <= w0 * 1e-4] = 0  # evanescent components only
    dq, dphi = np.gradient(q), phi[1] - phi[0]

    def direct(x, y):
        rho, th = np.hypot(x, y), np.arctan2(y, x)
        phase = np.exp(2j * np.pi * q[None, :] * rho * np.cos(phi[:, None] - th))
        return (spec * phase * q[None, :] * dq[None, :]).sum() * dphi

    pts = [(0, 0), (0.3, 0), (0, 0.3), (0.5, 0.5), (-0.8, 0.2), (0.1, -1.2), (1.5, 0.0), (0.0, 2.0)]
    c, dx = len(xy) // 2, xy[1] - xy[0]
    ref = np.array([direct(x, y) for x, y in pts])
    got = np.array([field[c + round(y / dx), c + round(x / dx)] for x, y in pts])
    scale = np.vdot(got, ref) / np.vdot(got, got)  # the FFT field is peak-normalised
    assert np.max(np.abs(ref - scale * got)) / np.abs(ref).max() < 2e-3
    # the field must be genuinely anisotropic, with x and y not interchangeable
    assert abs(field[c, c + 80] - field[c + 80, c]) > 1e-3 * abs(field[c, c])


def test_pole_wave_matches_fourier_field_for_a_known_pole():
    """Anisotropic synthetic rpp = exp(-(q/qc)^2)/(q - q_p(phi)): the Fourier field is the pole term plus a
    small background, so the two must agree in shape and phase (they differ only by the peak normalisation)."""
    from scipy.interpolate import RegularGridInterpolator
    from multilayer_atm.surface_wave import compute_pole_wave, compute_surface_wave

    phi = np.linspace(0.0, 2 * np.pi, 128, endpoint=False)
    k = np.linspace(200.0, 1.5e5, 900)
    qp = (2.0 + 0.08j) * (1 + 0.3 * np.cos(2 * phi))  # cycles/um
    qc = 6.0
    rp = np.exp(-((k[None, :] * 1e-4 / qc) ** 2)) / (k[None, :] * 1e-4 - qp[:, None])
    dinv = np.exp((qp / qc) ** 2) / 1e4  # d(1/rp)/dk in 1/cm^-1
    args = dict(w0_cm1=300.0, source_height_nm=25.0)
    xy, fp, _ = compute_pole_wave(phi, qp * 1e4, dinv, np.ones(128, bool), half_width_um=6.0, **args)
    xf, ff, _ = compute_surface_wave(phi, k, rp, fft_size=2048, **args)
    X, Y = np.meshgrid(xy, xy)
    ref = RegularGridInterpolator((xf, xf), ff, bounds_error=False)((Y, X))
    m = (np.hypot(X, Y) > 2.0) & (np.hypot(X, Y) < 5.5)
    scale = np.vdot(fp[m], ref[m]) / np.vdot(fp[m], fp[m])
    assert abs(np.angle(scale)) < 0.05  # prefactor phase of the residue term is right
    assert np.linalg.norm(ref[m] - scale * fp[m]) / np.linalg.norm(ref[m]) < 0.2


def test_pole_finder_on_isotropic_hbn_and_input_validation():
    from multilayer_atm.surface_wave import compute_pole_wave

    stack = StackSpec.from_layers([LayerSpec("vacuum", 0.0), LayerSpec("hBN", 100e-9), LayerSpec("aSiO2", 0.0)])
    phi, k, rp = solver.compute_isofreq_map(
        stack, w0=1500.0, kx_min=500.0, kx_max=25000.0, nk=128, nphi=32, workers=1, fast=True, return_complex=True
    )
    kp, dk, ok = solver.compute_isofreq_poles(stack, 1500.0, phi, k, rp)
    assert ok.all() and np.all(kp.imag > 0)
    assert np.ptp(kp.real) < 1e-3 * kp.real.mean()  # c axis out of plane: isotropic in plane
    xy, fp, _ = compute_pole_wave(phi, kp, dk, ok, w0_cm1=1500.0, half_width_um=3.0, n_pix=121)
    c = len(xy) // 2
    assert abs(fp[c, c + 40]) == pytest.approx(abs(fp[c + 40, c]), rel=0.03)
    with pytest.raises(ValueError, match="pole"):
        compute_pole_wave(phi, kp, dk, ok & False, w0_cm1=1500.0)
