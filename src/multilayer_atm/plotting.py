"""Static and interactive plotting helpers for the scientific workspace."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Polygon
import numpy as np
import streamlit as st

from .models import StackSpec


PALETTE = [
    "#084c61",
    "#177e89",
    "#49a078",
    "#f4c095",
    "#db504a",
    "#8f2d56",
    "#4f5d75",
    "#bfc0c0",
]

PSEUDO3D_SUBSTRATE_DISPLAY_UM = 0.16
PSEUDO3D_INTERIOR_SCALE = 0.28
PSEUDO3D_INTERIOR_MIN_UM = 0.015


def _is_dark_mode() -> bool:
    theme_ctx = getattr(st.context, "theme", None)
    return bool(getattr(theme_ctx, "base", "light") == "dark")


def _plot_chrome() -> dict[str, str]:
    """Return compact mode-aware UI colors used around rendered plots."""
    if _is_dark_mode():
        return {
            "figure_bg": "#111a26",
            "text": "#dbe5f1",
            "edge_dark": "#3f536d",
            "edge_light": "#5b7392",
            "shade_target": "#0c1420",
            "vacuum": "#85aacd",
            "marker_fill": "#e3edf8",
            "marker_edge": "#1f2c3d",
        }
    return {
        "figure_bg": "#f7fafd",
        "text": "#172534",
        "edge_dark": "#4b6178",
        "edge_light": "#364a62",
        "shade_target": "#1f2c3d",
        "vacuum": "#7da3c7",
        "marker_fill": "#fcf7e3",
        "marker_edge": "#2b3d51",
    }


def _rotate_uv(points: np.ndarray, alpha_deg: float) -> np.ndarray:
    ang = np.deg2rad(alpha_deg)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=float)
    return points @ rot.T


def _mix_color(base: str, target: str, weight: float) -> tuple[float, float, float]:
    b = np.asarray(mcolors.to_rgb(base), dtype=float)
    t = np.asarray(mcolors.to_rgb(target), dtype=float)
    return tuple((1.0 - weight) * b + weight * t)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec.copy()
    return vec / norm


def _face_normal(face3d: np.ndarray) -> np.ndarray:
    v1 = face3d[1] - face3d[0]
    v2 = face3d[2] - face3d[0]
    return _normalize(np.cross(v1, v2))


def _orient_face_outward(face3d: np.ndarray, slab_center: np.ndarray) -> np.ndarray:
    centered = np.mean(face3d, axis=0) - slab_center
    normal = _face_normal(face3d)
    if float(np.dot(normal, centered)) < 0.0:
        return face3d[::-1]
    return face3d


def _project_points(points3d: np.ndarray, v_to_x: float, v_to_y: float) -> np.ndarray:
    out = np.empty((points3d.shape[0], 2), dtype=float)
    out[:, 0] = points3d[:, 0] + v_to_x * points3d[:, 1]
    out[:, 1] = points3d[:, 2] + v_to_y * points3d[:, 1]
    return out


def _shade_side_color(base: str, normal: np.ndarray, shade_target: str) -> tuple[float, float, float]:
    light_dir = _normalize(np.array([0.34, -0.55, 1.0], dtype=float))
    lit = max(0.0, float(np.dot(normal, light_dir)))
    dark_weight = 0.46 - 0.24 * lit
    return _mix_color(base, shade_target, dark_weight)


def _is_vacuum_material(material: str) -> bool:
    m = material.strip().lower()
    return m in {"vac", "vacuum", "air"}


def plot_stack_pseudo3d(stack: StackSpec) -> plt.Figure:
    """Render a compact pseudo-3D stack preview for the current layer sequence."""
    indexed_layers = list(enumerate(stack.layers))
    if indexed_layers and _is_vacuum_material(str(indexed_layers[0][1].material)):
        indexed_layers = indexed_layers[1:]

    tvals = []
    for stack_index, layer in indexed_layers:
        if stack_index == len(stack.layers) - 1:
            tvals.append(PSEUDO3D_SUBSTRATE_DISPLAY_UM)
        else:
            tvals.append(max(layer.thickness_m * 1e6 * PSEUDO3D_INTERIOR_SCALE, PSEUDO3D_INTERIOR_MIN_UM))
    tvals_arr = np.asarray(tvals, dtype=float)

    chrome = _plot_chrome()
    fig_bg = chrome["figure_bg"]
    fg = chrome["text"]
    edge_dark = chrome["edge_dark"]
    edge_light = chrome["edge_light"]

    fig, ax = plt.subplots(figsize=(6.1, 3.9), dpi=120)
    fig.patch.set_facecolor(fig_bg)
    ax.set_facecolor(fig_bg)
    half_width = 0.5
    half_depth = 0.22
    v_to_x = 0.40
    v_to_y = 0.22
    local_corners = np.array(
        [
            [-half_width, -half_depth],
            [half_width, -half_depth],
            [half_width, half_depth],
            [-half_width, half_depth],
        ],
        dtype=float,
    )
    view_dir = _normalize(np.array([0.74, -1.0, 0.86], dtype=float))

    z_cursor = 0.0
    projected_points: list[np.ndarray] = []
    faces_to_draw: list[tuple[int, int, float, np.ndarray, tuple[float, float, float]]] = []
    indexed_thickness_layers = list(zip(indexed_layers, tvals_arr))
    for slab_order, ((stack_index, layer), th) in enumerate(reversed(indexed_thickness_layers)):
        z0 = z_cursor
        z1 = z_cursor + float(th)
        z_cursor = z1

        alpha_eff = float(layer.euler_deg[0])
        uv = _rotate_uv(local_corners, alpha_eff) + np.array([0.5, 0.0], dtype=float)

        bottom3d = np.column_stack([uv[:, 0], uv[:, 1], np.full(uv.shape[0], z0, dtype=float)])
        top3d = np.column_stack([uv[:, 0], uv[:, 1], np.full(uv.shape[0], z1, dtype=float)])
        slab_center = np.array([0.5, 0.0, 0.5 * (z0 + z1)], dtype=float)

        is_vac = _is_vacuum_material(str(layer.material))
        color = chrome["vacuum"] if is_vac else PALETTE[stack_index % len(PALETTE)]
        top_color = _mix_color(color, "#ffffff", 0.4 if is_vac else 0.25)
        faces3d = [
            (0, "top", _orient_face_outward(top3d.copy(), slab_center)),
            (1, "side", _orient_face_outward(np.vstack([bottom3d[0], bottom3d[1], top3d[1], top3d[0]]), slab_center)),
            (2, "side", _orient_face_outward(np.vstack([bottom3d[1], bottom3d[2], top3d[2], top3d[1]]), slab_center)),
            (3, "side", _orient_face_outward(np.vstack([bottom3d[2], bottom3d[3], top3d[3], top3d[2]]), slab_center)),
            (4, "side", _orient_face_outward(np.vstack([bottom3d[3], bottom3d[0], top3d[0], top3d[3]]), slab_center)),
        ]

        for face_rank, face_kind, face3d in faces3d:
            normal = _face_normal(face3d)
            if face_kind == "side" and float(np.dot(normal, view_dir)) <= 1e-9:
                continue

            projected = _project_points(face3d, v_to_x, v_to_y)
            projected_points.extend(projected)
            depth = float(np.dot(np.mean(face3d, axis=0), view_dir))
            face_color = top_color if face_kind == "top" else _shade_side_color(color, normal, chrome["shade_target"])
            face_priority = 1 if face_kind == "side" else 2
            faces_to_draw.append((slab_order, face_priority, depth, projected, face_color))

    faces_to_draw.sort(key=lambda rec: (rec[0], rec[1], rec[2]))
    for _, _, _, poly2d, face_color in faces_to_draw:
        edge_color = edge_dark if float(np.mean(face_color)) > 0.72 else edge_light
        ax.add_patch(
            Polygon(
                poly2d,
                closed=True,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=0.78,
                alpha=0.96,
                joinstyle="round",
            )
        )

    if projected_points:
        all_pts = np.asarray(projected_points, dtype=float)
        xmin = float(np.min(all_pts[:, 0]) - 0.08)
        xmax = float(np.max(all_pts[:, 0]) + 0.08)
        ymin = float(np.min(all_pts[:, 1]) - 0.08)
        ymax = float(np.max(all_pts[:, 1]) + 0.08)
    else:
        xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Pseudo-3D stack view", fontsize=11, weight="bold", color=fg)
    for spine in ax.spines.values():
        spine.set_color(edge_light)
    fig.tight_layout()
    return fig


def plot_heatmap(
    x: Sequence[float],
    y: Sequence[float],
    z: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    cmap: str = "magma",
) -> plt.Figure:
    """Render a static Matplotlib dispersion heatmap for export."""
    xx = np.asarray(x)
    yy = np.asarray(y)

    fig, ax = plt.subplots(figsize=(8.8, 5.5), dpi=120)
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        cmap=cmap,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Im(rpp)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, weight="bold")
    fig.tight_layout()
    return fig


def plot_heatmap_interactive(
    x: Sequence[float],
    y: Sequence[float],
    z: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    cmap: str = "Magma",
    height: int = 560,
    peak_overlay: dict[str, np.ndarray] | None = None,
):
    """Render an interactive Plotly dispersion heatmap for the Streamlit UI."""
    import plotly.graph_objects as go

    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    zz = np.asarray(z, dtype=float)
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=xx,
                y=yy,
                z=zz,
                colorscale=cmap,
                colorbar=dict(
                    title="Im(rpp)",
                    x=1.03,
                    y=0.5,
                    len=0.88,
                    thickness=14,
                ),
            )
        ]
    )
    chrome = _plot_chrome()
    if peak_overlay is not None and len(peak_overlay.get("x", [])) > 0:
        fig.add_trace(
            go.Scatter(
                x=np.asarray(peak_overlay["x"], dtype=float),
                y=np.asarray(peak_overlay["y"], dtype=float),
                mode="markers",
                marker=dict(size=5, color=chrome["marker_fill"], line=dict(color=chrome["marker_edge"], width=0.8)),
                name="Peak dots",
                hovertemplate="w=%{y:.3f}<br>kx=%{x:.3f}<br>row max Im(rpp)=%{customdata:.4g}<extra></extra>",
                customdata=np.asarray(peak_overlay["value"], dtype=float),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_dark" if _is_dark_mode() else "plotly_white",
        height=int(height),
        margin=dict(l=20, r=72, t=55, b=20),
    )
    return fig


def _linear_edges(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        step = max(abs(values[0]) * 0.1, 1.0)
        return np.array([values[0] - 0.5 * step, values[0] + 0.5 * step], dtype=float)
    step = float(np.mean(np.diff(values)))
    return np.concatenate(([values[0] - 0.5 * step], values + 0.5 * step))


def _periodic_edges(values: np.ndarray, period: float) -> np.ndarray:
    if values.size == 1:
        return np.array([values[0] - 0.5 * period, values[0] + 0.5 * period], dtype=float)
    step = float(np.mean(np.diff(values)))
    return np.concatenate(([values[0] - 0.5 * step], values + 0.5 * step))


def plot_polar_isofrequency(
    phi_rad: Sequence[float],
    kx: Sequence[float],
    z: np.ndarray,
    title: str,
    cmap: str = "cividis",
) -> plt.Figure:
    """Render a static Matplotlib polar isofrequency map for export."""
    phi = np.asarray(phi_rad, dtype=float)
    radius = np.asarray(kx, dtype=float)
    theta_edges = _periodic_edges(phi, 2.0 * np.pi)
    r_edges = _linear_edges(radius)
    theta_grid, r_grid = np.meshgrid(theta_edges, r_edges, indexing="ij")

    fig = plt.figure(figsize=(7.4, 7.0), dpi=120)
    ax = fig.add_subplot(111, projection="polar")
    mesh = ax.pcolormesh(theta_grid, r_grid, z, shading="auto", cmap=cmap)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.12)
    cbar.set_label("Im(rpp)")
    ax.set_title(title, fontsize=12, weight="bold", pad=16)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlim(0.0, radius.max())
    ax.grid(alpha=0.45)
    ax.set_rlabel_position(112)
    fig.tight_layout()
    return fig


def plot_polar_isofrequency_interactive(
    phi_rad: Sequence[float],
    kx: Sequence[float],
    z: np.ndarray,
    title: str,
    cmap: str = "Cividis",
    radial_label: str = "kx (cm^-1)",
    height: int = 640,
    peak_overlay: dict[str, np.ndarray] | None = None,
):
    """Render an interactive Plotly polar isofrequency map for the Streamlit UI."""
    import plotly.graph_objects as go

    phi = np.asarray(phi_rad, dtype=float)
    radius = np.asarray(kx, dtype=float)
    values = np.asarray(z, dtype=float)

    if phi.size == 0 or radius.size == 0:
        return go.Figure()

    if phi.size > 1:
        dtheta = float(np.mean(np.diff(phi)))
    else:
        dtheta = 2.0 * np.pi

    theta_deg = np.rad2deg(phi)
    width_deg = np.full(phi.shape[0], np.rad2deg(dtheta), dtype=float)

    if radius.size > 1:
        dr = float(np.mean(np.diff(radius)))
    else:
        dr = max(abs(float(radius[0])) * 0.2, 1.0)

    bar_r = np.repeat(radius, phi.size)
    bar_theta = np.tile(theta_deg, radius.size)
    bar_width = np.tile(width_deg, radius.size)
    bar_color = values.T.reshape(-1)
    bar_base = np.maximum(bar_r - 0.5 * dr, 0.0)

    fig = go.Figure(
        data=[
            go.Barpolar(
                r=bar_r,
                theta=bar_theta,
                width=bar_width,
                marker=dict(
                    color=bar_color,
                    colorscale=cmap,
                    colorbar=dict(
                        title="Im(rpp)",
                        x=1.03,
                        y=0.5,
                        len=0.88,
                        thickness=14,
                    ),
                    line=dict(width=0),
                ),
                base=bar_base,
                opacity=0.98,
                hovertemplate="phi=%{theta:.1f} deg<br>kx=%{r:.3f}<br>Im(rpp)=%{marker.color:.4g}<extra></extra>",
            )
        ]
    )
    chrome = _plot_chrome()
    if peak_overlay is not None and len(peak_overlay.get("r", [])) > 0:
        fig.add_trace(
            go.Scatterpolar(
                r=np.asarray(peak_overlay["r"], dtype=float),
                theta=np.asarray(peak_overlay["theta_deg"], dtype=float),
                mode="markers",
                marker=dict(size=5, color=chrome["marker_fill"], line=dict(color=chrome["marker_edge"], width=0.8)),
                name="Peak dots",
                customdata=np.asarray(peak_overlay["value"], dtype=float),
                hovertemplate="phi=%{theta:.1f} deg<br>kx=%{r:.3f}<br>row max Im(rpp)=%{customdata:.4g}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        template="plotly_dark" if _is_dark_mode() else "plotly_white",
        height=int(height),
        margin=dict(l=20, r=72, t=55, b=20),
        polar=dict(
            radialaxis=dict(range=[0.0, float(radius.max())], title=radial_label),
            angularaxis=dict(direction="counterclockwise", rotation=0),
        ),
    )
    return fig
