"""Shared page styling and top-level layout helpers."""

from __future__ import annotations

import streamlit as st


def configure_page() -> None:
    """Set the Streamlit page metadata and use the wide scientific workspace layout."""
    st.set_page_config(
        page_title="Passler rpp Explorer",
        page_icon=":material/waves:",
        layout="wide",
    )


def apply_theme() -> None:
    """Apply minimal scoped layout helpers that complement config.toml theming."""
    st.markdown(
        """
<style>
/* Keep layout compact without overriding Streamlit theme colors/fonts. */
.block-container {
    max-width: min(98vw, 1900px);
    padding-top: 2.2rem;
    padding-bottom: 0.28rem;
    padding-left: clamp(0.55rem, 1vw, 0.9rem);
    padding-right: clamp(0.55rem, 1vw, 0.9rem);
}

.topbar {
    padding: 0.2rem 0 0.18rem 0;
    margin-bottom: 0.08rem;
}

.topbar h1 {
    margin-bottom: 0.1rem;
}

.stack-sequence {
    display: flex;
    flex-direction: column;
    gap: 0.18rem;
    margin-top: 0.14rem;
}

@media (max-width: 980px) {
    .block-container {
        padding-top: 1.9rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_top_bar(status: dict[str, str] | None = None) -> None:
    """Render the compact top bar with run context and freshness information."""
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    with st.container(gap=None, horizontal_alignment="distribute"):
        title_col, worker_col = st.columns([0.82, 0.18], gap=None, vertical_alignment="bottom")
        with title_col:
            st.title("Multilayer anisotropic transfer matrix", anchor=False)
            st.caption("Developed by [Edoardo Vicentini](https://github.com/vicentini-edoardo)")
        with worker_col:
            with st.container():
                label_col, slider_col = st.columns([0.36, 0.64], gap=None, vertical_alignment="center")
                with label_col:
                    st.caption("CPU workers")
                with slider_col:
                    st.slider(
                        "CPU workers",
                        min_value=1,
                        max_value=16,
                        step=1,
                        key="worker_count",
                        label_visibility="collapsed",
                    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_footer() -> None:
    """Render a compact scientific-units footer for the workspace."""
    st.caption(
        "Units: w in cm⁻¹, kx UI in 10^3 cm⁻¹, boundary layers semi-infinite, Euler angles per layer."
    )
    st.caption(
        "Reference: N. C. Passler and A. Paarmann, Journal of the Optical Society of America B 34, 2128 (2017), "
        "DOI: https://doi.org/10.1364/JOSAB.34.002128"
    )
    st.caption(
        "pyGTM repository: https://github.com/pyMatJ/pyGTM"
    )
