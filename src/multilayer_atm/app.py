"""Compose the Streamlit workspace from package-local modules."""

from __future__ import annotations

import streamlit as st

from .materials import material_catalog, material_notes
from .presets import SPEED_PRESETS
from .ui.calculation_views import (
    init_calculation_state,
    render_mode_input_strip,
    render_wave_options,
    render_results_panel,
    render_run_controls_panel,
    workspace_status,
)
from .ui.layer_builder import init_layer_state, render_stack_panel
from .ui.theme_layout import apply_theme, configure_page, render_footer, render_top_bar


def run_app() -> None:
    """Render the full Streamlit application."""
    configure_page()
    apply_theme()
    init_layer_state()
    init_calculation_state(SPEED_PRESETS)

    catalog = list(material_catalog())
    notes = material_notes()

    render_top_bar(workspace_status())

    # 1:2:1 layout: layers | plots | run controls. The right column is filled first
    # so a Calculate click updates session state before the plots render.
    stack_col, plot_col, run_col = st.columns([1, 2, 1], gap="small", vertical_alignment="top")

    with run_col:
        st.subheader(":material/tune: Run controls", anchor=False)
        with st.container(border=True):
            render_run_controls_panel(
                speed_presets=SPEED_PRESETS,
                between=lambda: render_mode_input_strip(speed_presets=SPEED_PRESETS),
            )
        st.subheader(":material/waves: Real-space wave", anchor=False)
        with st.container(border=True):
            render_wave_options()

    with stack_col:
        st.subheader(":material/layers: Layer sequence", anchor=False)
        with st.container(gap=None):
            render_stack_panel(catalog=catalog, notes=notes, show_header=False)

    with plot_col:
        render_results_panel()

    render_footer()
