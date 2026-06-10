from __future__ import annotations

import numpy as np
import streamlit as st

from multilayer_atm.plotting import (
    plot_heatmap,
    plot_heatmap_interactive,
    plot_polar_isofrequency,
    plot_polar_isofrequency_interactive,
)
from multilayer_atm.ui import calculation_views as cv


def setup_function() -> None:
    st.session_state.clear()


def test_plot_heatmap_applies_manual_color_limits() -> None:
    fig = plot_heatmap(
        x=[0.0, 1.0],
        y=[2.0, 3.0],
        z=np.array([[0.2, 0.4], [0.6, 0.8]], dtype=float),
        xlabel="x",
        ylabel="y",
        title="map",
        vmin=0.1,
        vmax=0.9,
    )

    image = fig.axes[0].images[0]

    assert image.get_clim() == (0.1, 0.9)


def test_plot_heatmap_interactive_applies_manual_color_limits() -> None:
    fig = plot_heatmap_interactive(
        x=[0.0, 1.0],
        y=[2.0, 3.0],
        z=np.array([[0.2, 0.4], [0.6, 0.8]], dtype=float),
        xlabel="x",
        ylabel="y",
        title="map",
        zmin=0.1,
        zmax=0.9,
    )

    trace = fig.data[0]

    assert trace.zmin == 0.1
    assert trace.zmax == 0.9


def test_plot_polar_isofrequency_applies_manual_color_limits() -> None:
    fig = plot_polar_isofrequency(
        phi_rad=[0.0, np.pi / 2.0],
        kx=[1.0, 2.0],
        z=np.array([[0.2, 0.4], [0.6, 0.8]], dtype=float),
        title="iso",
        vmin=0.1,
        vmax=0.9,
    )

    mesh = fig.axes[0].collections[0]

    assert mesh.get_clim() == (0.1, 0.9)


def test_plot_polar_isofrequency_interactive_applies_manual_color_limits() -> None:
    fig = plot_polar_isofrequency_interactive(
        phi_rad=[0.0, np.pi / 2.0],
        kx=[1.0, 2.0],
        z=np.array([[0.2, 0.4], [0.6, 0.8]], dtype=float),
        title="iso",
        cmin=0.1,
        cmax=0.9,
    )

    marker = fig.data[0].marker

    assert marker.cmin == 0.1
    assert marker.cmax == 0.9


def test_active_plot_color_limits_return_none_in_auto_mode() -> None:
    st.session_state.plot_manual_color_limits = False
    st.session_state.plot_color_min = 0.1
    st.session_state.plot_color_max = 0.9

    assert cv._active_plot_color_limits() == (None, None)


def test_active_plot_color_limits_return_values_when_valid() -> None:
    st.session_state.plot_manual_color_limits = True
    st.session_state.plot_color_min = 0.1
    st.session_state.plot_color_max = 0.9

    assert cv._active_plot_color_limits() == (0.1, 0.9)


def test_active_plot_color_limits_return_none_when_invalid() -> None:
    st.session_state.plot_manual_color_limits = True
    st.session_state.plot_color_min = 0.9
    st.session_state.plot_color_max = 0.1

    assert cv._active_plot_color_limits() == (None, None)


def test_seed_manual_color_limits_from_result_uses_finite_data() -> None:
    st.session_state.plot_color_min = -1.0
    st.session_state.plot_color_max = -1.0

    cv._seed_manual_color_limits(np.array([[np.nan, 0.4], [0.6, np.inf]], dtype=float))

    assert st.session_state.plot_color_min == 0.4
    assert st.session_state.plot_color_max == 0.6


def test_plot_image_cache_key_changes_with_manual_color_limits(monkeypatch) -> None:
    st.session_state.calc_mode = "Im(rpp) as f(w, kx)"
    st.session_state.plot_colormap = "Magma"
    st.session_state.selected_map_history_id = "map_1"
    st.session_state.map_result = (
        np.array([1.0, 2.0], dtype=float),
        np.array([3.0, 4.0], dtype=float),
        np.array([[0.2, 0.4], [0.6, 0.8]], dtype=float),
    )
    st.session_state.plot_manual_color_limits = True
    st.session_state.plot_color_min = 0.1
    st.session_state.plot_color_max = 0.9

    calls: list[tuple[object, ...]] = []

    def _fake_cached_bytes(cache_key: tuple[object, ...], builder):
        calls.append(cache_key)
        return b"png"

    monkeypatch.setattr(cv, "_cached_bytes", _fake_cached_bytes)
    monkeypatch.setattr(cv, "_stack_plot_filename_stem", lambda mode, stack=None: "plot")

    cv._current_plot_image_payload()
    st.session_state.plot_color_max = 1.1
    cv._current_plot_image_payload()

    assert len(calls) == 2
    assert calls[0] != calls[1]
