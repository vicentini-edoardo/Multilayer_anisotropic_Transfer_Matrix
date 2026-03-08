"""Compose the Streamlit workspace from package-local modules."""

from __future__ import annotations

import streamlit as st

from .materials import material_catalog, material_notes
from .presets import SPEED_PRESETS
from .ui.calculation_views import init_calculation_state, render_calculation_panel, render_results_panel, workspace_status
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

    # Keep the three core research tasks visible at once: build the stack,
    # configure the calculation, and inspect the result.
    stack_col, calc_col, results_col = st.columns([0.26, 0.30, 0.44], gap="small", vertical_alignment="top")

    with stack_col:
        render_stack_panel(catalog=catalog, notes=notes)

    with calc_col:
        render_calculation_panel(speed_presets=SPEED_PRESETS)

    with results_col:
        render_results_panel()

    render_footer()
