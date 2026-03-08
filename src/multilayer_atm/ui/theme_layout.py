"""Shared page styling and top-level layout helpers."""

from __future__ import annotations

from typing import Mapping, Sequence

import streamlit as st


def configure_page() -> None:
    """Set the Streamlit page metadata and use the wide scientific workspace layout."""
    st.set_page_config(
        page_title="Passler rpp Explorer",
        page_icon=":material/waves:",
        layout="wide",
    )


def single_choice(label: str, options: Sequence[str], default: str, key: str) -> str:
    """Render a single-choice control with stable keyed state across reruns."""
    options_list = list(options)
    resolved_default = default if default in options_list else options_list[0]
    current_value = st.session_state.get(key)
    if current_value not in options_list:
        st.session_state[key] = resolved_default
    if hasattr(st, "segmented_control"):
        selected = st.segmented_control(label, options_list, key=key)
        return str(selected if selected is not None else st.session_state[key])
    return str(st.radio(label, options_list, horizontal=True, key=key))


def _status_badge(label: str, value: str, tone: str = "blue") -> str:
    return f":{tone}-badge[{label}: {value}]"


def apply_theme() -> None:
    """Apply the shared light scientific theme used across the workspace."""
    st.markdown(
        """
<style>
:root {
    --workspace-bg: #f4f7fb;
    --surface: #fbfcfe;
    --surface-2: #f7f9fc;
    --surface-3: #eef3f9;
    --ink: #122033;
    --ink-muted: #566579;
    --border: #e4eaf2;
    --accent: #2457d6;
    --accent-2: #3f6fe3;
    --accent-soft: #e9f0ff;
    --danger-soft: #fff0ef;
    --success-soft: #edf8ef;
    --warning-soft: #fff8e8;
    --shadow: 0 18px 40px -30px rgba(18, 32, 51, 0.4);
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    color: var(--ink) !important;
    background: var(--workspace-bg) !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top right, rgba(36, 87, 214, 0.08), transparent 24%),
        linear-gradient(180deg, #f8fbff 0%, var(--workspace-bg) 22%, var(--workspace-bg) 100%) !important;
}

.block-container {
    max-width: min(98vw, 1900px);
    padding-top: 0.35rem;
    padding-bottom: 0.45rem;
    padding-left: clamp(0.55rem, 1vw, 0.9rem);
    padding-right: clamp(0.55rem, 1vw, 0.9rem);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--ink) !important;
    letter-spacing: 0.01em;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"] *,
[data-testid="stWidgetLabel"] * {
    color: var(--ink) !important;
}

[data-baseweb="input"] input,
[data-baseweb="select"] input,
[data-baseweb="textarea"] textarea {
    color: var(--ink) !important;
    background: white !important;
}

.topbar {
    border-bottom: 1px solid color-mix(in srgb, var(--border) 55%, transparent);
    padding: 0.15rem 0 0.35rem 0;
    margin-bottom: 0.3rem;
}

.topbar-title {
    margin: 0;
    font-size: clamp(0.92rem, 1.25vw, 1.05rem);
    line-height: 1.1;
    font-weight: 600;
}

.topbar-subtitle {
    margin: 0.08rem 0 0 0;
    color: var(--ink-muted) !important;
    font-size: 0.68rem;
}

.section-label {
    margin: 0 0 0.08rem 0;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-size: 0.58rem;
    font-weight: 700;
    color: var(--ink-muted) !important;
}

.muted-note {
    color: var(--ink-muted) !important;
    font-size: 0.72rem;
}

.stack-sequence {
    display: flex;
    flex-direction: column;
    gap: 0.28rem;
    margin-top: 0.2rem;
}

.stack-sequence-item {
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    padding: 0.44rem 0.56rem;
}

.stack-sequence-item.selected {
    border-color: rgba(36, 87, 214, 0.72);
    box-shadow: inset 0 0 0 1px rgba(36, 87, 214, 0.22);
    background: linear-gradient(180deg, rgba(233, 240, 255, 0.95), rgba(250, 252, 255, 1));
}

.stack-role {
    display: inline-block;
    padding: 0.08rem 0.35rem;
    border-radius: 999px;
    background: var(--surface-3);
    color: var(--ink-muted);
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.stack-role.selected {
    background: var(--accent-soft);
    color: var(--accent);
}

.result-shell {
    min-height: 500px;
}

.result-toolbar-note {
    color: var(--ink-muted) !important;
    font-size: 0.68rem;
    margin-top: 0.08rem;
}

.status-pill {
    border: 1px solid color-mix(in srgb, var(--border) 55%, transparent);
    border-radius: 999px;
    padding: 0.1rem 0.38rem;
    background: white;
    font-size: 0.64rem;
    color: var(--ink-muted);
    display: inline-block;
}

.status-pill.current {
    background: var(--success-soft);
    border-color: #cfe6d4;
    color: #1f6a35;
}

.status-pill.stale {
    background: var(--warning-soft);
    border-color: #ead8a5;
    color: #8a6512;
}

[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button {
    border-radius: 8px;
    border: 1px solid color-mix(in srgb, var(--border) 45%, transparent);
    min-height: 2rem;
    padding-top: 0.2rem;
    padding-bottom: 0.2rem;
    font-size: 0.78rem;
    text-align: center;
    line-height: 1.1;
    box-shadow: none;
}

[data-testid="stButton"] button p,
[data-testid="stDownloadButton"] button p {
    text-align: center !important;
    margin: 0 !important;
}

.stack-sequence [data-testid="stButton"] button {
    text-align: left;
    line-height: 1.22;
    padding-left: 0.7rem;
    padding-right: 0.7rem;
}

.stack-sequence [data-testid="stButton"] button p {
    text-align: left !important;
}

.stack-sequence [data-testid="stButton"]:has(strong) button,
[data-testid="stButton"]:has(strong) button {
    text-align: left;
    line-height: 1.22;
    padding-left: 0.7rem;
    padding-right: 0.7rem;
}

.stack-sequence [data-testid="stButton"]:has(strong) button p,
[data-testid="stButton"]:has(strong) button p {
    text-align: left !important;
}

.stack-sequence [data-testid="stButton"] button:has(strong),
.stack-sequence [data-testid="stButton"] button[kind="primary"],
.stack-sequence [data-testid="stButton"] button[data-testid="baseButton-primary"] {
    min-height: 2.55rem;
    padding-top: 0.42rem;
    padding-bottom: 0.42rem;
    padding-left: 0.95rem;
    padding-right: 0.95rem;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    font-size: 0.84rem;
    font-weight: 600;
    border-width: 1.2px;
    transition: background 120ms ease, border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
}

[data-testid="stButton"]:has(strong) button,
[data-testid="stButton"]:has(strong) button[kind="primary"],
[data-testid="stButton"]:has(strong) button[data-testid="baseButton-primary"] {
    min-height: 2.55rem;
    padding-top: 0.42rem;
    padding-bottom: 0.42rem;
    padding-left: 0.95rem;
    padding-right: 0.95rem;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    font-size: 0.84rem;
    font-weight: 600;
    border-width: 1.2px;
    transition: background 120ms ease, border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
}

.stack-sequence [data-testid="stButton"] button:has(strong):not([kind="primary"]):not([data-testid="baseButton-primary"]) {
    background: linear-gradient(180deg, #fff0df 0%, #ffe4c5 100%) !important;
    border-color: #e0b171 !important;
    color: #5a3510 !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.92),
        0 8px 16px -16px rgba(121, 72, 14, 0.42);
}

[data-testid="stButton"]:has(strong) button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {
    background: linear-gradient(180deg, #fff0df 0%, #ffe4c5 100%) !important;
    border-color: #e0b171 !important;
    color: #5a3510 !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.92),
        0 8px 16px -16px rgba(121, 72, 14, 0.42);
}

.stack-sequence [data-testid="stButton"] button:has(strong):not([kind="primary"]):not([data-testid="baseButton-primary"]):hover {
    background: linear-gradient(180deg, #ffe8cf 0%, #ffdcb5 100%) !important;
    border-color: #cf9449 !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.95),
        0 10px 22px -18px rgba(121, 72, 14, 0.55);
    transform: translateY(-0.5px);
}

[data-testid="stButton"]:has(strong) button:not([kind="primary"]):not([data-testid="baseButton-primary"]):hover {
    background: linear-gradient(180deg, #ffe8cf 0%, #ffdcb5 100%) !important;
    border-color: #cf9449 !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.95),
        0 10px 22px -18px rgba(121, 72, 14, 0.55);
    transform: translateY(-0.5px);
}

.stack-sequence [data-testid="stButton"] button:has(strong) p,
.stack-sequence [data-testid="stButton"] button[kind="primary"] p {
    line-height: 1.3 !important;
}

[data-testid="stButton"]:has(strong) button p {
    line-height: 1.3 !important;
}

.stack-sequence [data-testid="stButton"] button:has(strong) strong,
.stack-sequence [data-testid="stButton"] button[kind="primary"] strong,
.stack-sequence [data-testid="stButton"] button[data-testid="baseButton-primary"] strong {
    display: inline-block;
    margin-bottom: 0.14rem;
    font-size: 0.89rem;
    font-weight: 750;
    letter-spacing: 0.012em;
}

[data-testid="stButton"]:has(strong) button strong {
    display: inline-block;
    margin-bottom: 0.14rem;
    font-size: 0.89rem;
    font-weight: 750;
    letter-spacing: 0.012em;
}

.stack-sequence [data-testid="stButton"] button:has(strong) p:not(:has(strong)) {
    color: color-mix(in srgb, var(--ink) 72%, #8a5b27) !important;
}

[data-testid="stButton"]:has(strong) button p:not(:has(strong)) {
    color: color-mix(in srgb, var(--ink) 72%, #8a5b27) !important;
}

.stack-sequence [data-testid="stButton"] button:not(:has(p br)) {
    min-height: 2rem;
    padding-left: 0.2rem;
    padding-right: 0.2rem;
}

[data-testid="stButton"] button[kind="primary"],
[data-testid="stButton"] button[data-testid="baseButton-primary"],
[data-testid="stDownloadButton"] button[kind="primary"] {
    border-color: rgba(36, 87, 214, 0.6);
    background: linear-gradient(180deg, var(--accent-2), var(--accent));
    color: white !important;
    box-shadow: 0 10px 24px -20px rgba(36, 87, 214, 0.95);
}

.stack-sequence [data-testid="stButton"] button[kind="primary"],
.stack-sequence [data-testid="stButton"] button[data-testid="baseButton-primary"] {
    border-color: #b86c1c !important;
    background: linear-gradient(180deg, #e89a42 0%, #d77f25 58%, #bf6817 100%) !important;
    color: white !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.16),
        0 14px 26px -20px rgba(137, 78, 15, 0.85);
}

[data-testid="stButton"]:has(strong) button[kind="primary"],
[data-testid="stButton"]:has(strong) button[data-testid="baseButton-primary"] {
    border-color: #b86c1c !important;
    background: linear-gradient(180deg, #e89a42 0%, #d77f25 58%, #bf6817 100%) !important;
    color: white !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.16),
        0 14px 26px -20px rgba(137, 78, 15, 0.85);
}

[data-testid="stButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {
    border-color: color-mix(in srgb, var(--accent) 38%, transparent);
}

[data-testid="stTabs"] button {
    border-radius: 8px 8px 0 0;
    min-height: 2rem;
    padding-top: 0.18rem;
    padding-bottom: 0.18rem;
    font-size: 0.76rem;
}

[data-testid="stExpander"] {
    border: 1px solid color-mix(in srgb, var(--border) 45%, transparent);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.58);
}

[data-testid="stExpander"] details summary {
    padding-top: 0.1rem;
    padding-bottom: 0.1rem;
}

[data-testid="stVerticalBlock"] {
    gap: 0.35rem;
}

[data-testid="stHorizontalBlock"] {
    gap: 0.35rem;
}

[data-baseweb="input"] input,
[data-baseweb="select"] input,
[data-baseweb="base-input"] input {
    min-height: 2rem !important;
    font-size: 0.8rem !important;
}

[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stToggle"] label {
    font-size: 0.76rem !important;
}

[data-testid="stCaptionContainer"] {
    margin-top: -0.12rem;
}

[data-testid="stMarkdownContainer"] p {
    margin-bottom: 0.2rem;
}

[data-testid="stTabs"] [role="tablist"] {
    gap: 0.2rem;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    font-weight: 600;
}

[data-testid="stInfo"] {
    padding-top: 0.45rem;
    padding-bottom: 0.45rem;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
}

@media (max-width: 980px) {
    .block-container {
        padding-left: 0.55rem;
        padding-right: 0.55rem;
    }
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_top_bar(status: Mapping[str, str]) -> None:
    """Render the compact top bar with run context and freshness information."""
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    left, right = st.columns([0.62, 0.38], gap="small", vertical_alignment="center")
    with left:
        st.markdown('<p class="topbar-title">Multilayer anisotropic transfer matrix</p>', unsafe_allow_html=True)
        st.markdown('<p class="topbar-subtitle">Stack design, setup, and results.</p>', unsafe_allow_html=True)
    with right:
        badges = " ".join([_status_badge("Preset", status.get("preset", "Normal")), _status_badge("Workers", status.get("workers", "4"), tone="green")])
        st.markdown(badges)
        stale_class = "current" if status.get("freshness") == "Current" else "stale"
        st.markdown(
            f'<div style="margin-top:0.18rem;"><span class="status-pill {stale_class}">'
            f'{status.get("mode", "Im(rpp)")} • {status.get("freshness", "Stale")}'
            f"</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_footer() -> None:
    """Render a compact scientific-units footer for the workspace."""
    st.caption(
        "Units: w in cm⁻¹, kx UI in 10^3 cm⁻¹, boundary layers semi-infinite, Euler optional."
    )
