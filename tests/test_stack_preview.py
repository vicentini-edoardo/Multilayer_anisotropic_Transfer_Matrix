from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from multilayer_atm.models import LayerSpec, StackSpec
from multilayer_atm.plotting import plot_stack_pseudo3d


def test_stack_preview_keeps_thin_layers_proportional() -> None:
    stack = StackSpec.from_layers([
        LayerSpec('vacuum', 0), LayerSpec('thin', 10e-9),
        LayerSpec('thick', 40e-9), LayerSpec('substrate', 0),
    ])
    fig = plot_stack_pseudo3d(stack)
    # Each unrotated slab has two visible side faces followed by its top.
    sides = fig.axes[0].patches
    def height(patch):
        points = patch.get_xy()
        return abs(points[2, 1] - points[1, 1])
    assert np.isclose(height(sides[6]) / height(sides[3]), 0.25)
    plt.close(fig)


def test_repeated_materials_share_color() -> None:
    stack = StackSpec.from_layers([
        LayerSpec('vacuum', 0), LayerSpec('same', 100e-9),
        LayerSpec('other', 100e-9), LayerSpec('same', 100e-9),
        LayerSpec('substrate', 0),
    ])
    fig = plot_stack_pseudo3d(stack)
    patches = fig.axes[0].patches
    assert np.allclose(patches[5].get_facecolor(), patches[11].get_facecolor())
    assert not np.allclose(patches[5].get_facecolor(), patches[8].get_facecolor())
    plt.close(fig)
