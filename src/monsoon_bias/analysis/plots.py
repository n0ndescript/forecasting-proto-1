"""Publication-quality plots.

All figures are saved as 300-DPI PNGs into :data:`config.FIGURES_DIR`.
Maps use cartopy (PlateCarree) with coastlines and Indian state
boundaries. Colormaps come from cmocean (``rain`` for sequential
rainfall fields; ``balance`` for diverging bias maps).
"""

from __future__ import annotations

from pathlib import Path

from .. import config


def plot_mean_bias_map(bias_map, output_path: Path | None = None):
    """Diverging colormap centered at 0 (cmocean.cm.balance)."""
    raise NotImplementedError


def plot_rmse_map(rmse_map, output_path: Path | None = None):
    """Sequential colormap (cmocean.cm.amp or .rain)."""
    raise NotImplementedError


def plot_bias_by_bsiso(bias_by_phase, output_path: Path | None = None):
    """Small-multiples: 2x4 panel of bias maps, one per BSISO phase, with
    a shared color scale.
    """
    raise NotImplementedError


def plot_bias_vs_elevation(bias_da, elevation_da, output_path: Path | None = None):
    """Scatter of pointwise bias vs elevation with a regression line."""
    raise NotImplementedError


def plot_bias_by_region(region_stats, output_path: Path | None = None):
    """Bar chart of mean bias and RMSE per region."""
    raise NotImplementedError


def plot_bias_vs_rainfall(bias_by_mag, output_path: Path | None = None):
    """Mean bias as a function of observed-rainfall bin."""
    raise NotImplementedError
