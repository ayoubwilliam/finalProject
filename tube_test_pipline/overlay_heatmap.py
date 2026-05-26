"""
Shared heatmap overlay utilities.
Provides the Red-Clear-Green colormap and reusable plotting helpers for
both data generation heatmaps and evaluation visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def build_overlay_colormap():
    """Builds a colormap: Green (negative) → Transparent (zero) → Red (positive)."""
    colors = [
        (0.0, (0, 1, 0, 1.0)),
        (0.5, (0, 1, 0, 0.0)),
        (0.5, (1, 0, 0, 0.0)),
        (1.0, (1, 0, 0, 1.0)),
    ]
    return LinearSegmentedColormap.from_list("RedClearGreen", colors, N=256)


def add_colorbar(fig, ax, im):
    """Appends a scaled colorbar to the given axis."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)


def plot_base_image(ax, base_img, title, title_fontsize=14):
    """Plots a standard grayscale base image without an overlay."""
    ax.imshow(base_img, cmap="gray")
    ax.set_title(title, fontsize=title_fontsize)
    ax.axis("off")


def plot_heatmap_overlay(ax, base_img, overlay_img, title, cmap, fig, title_fontsize=14):
    """Plots a continuous heatmap over a grayscale base image with a symmetric colorbar."""
    ax.imshow(base_img, cmap="gray")
    limit = max(np.max(np.abs(overlay_img)), 1e-5)
    im = ax.imshow(overlay_img, cmap=cmap, alpha=0.9, vmin=-limit, vmax=limit)
    ax.set_title(title, fontsize=title_fontsize)
    ax.axis("off")
    add_colorbar(fig, ax, im)
