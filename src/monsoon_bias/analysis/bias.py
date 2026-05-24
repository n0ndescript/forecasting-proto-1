"""Bias diagnostics.

All inputs come from the master Zarr store (see
:mod:`monsoon_bias.processing.store`). Outputs are saved alongside as
small NetCDFs under :data:`config.OUTPUTS_DIR`.

Diagnostics produced (all on the common 0.25° grid unless noted):
    - mean_bias_map(lat, lon)
    - rmse_map(lat, lon)
    - bias_by_bsiso_phase(phase, lat, lon)        # 0..8 (0 = inactive)
    - bias_by_elevation(elev_bin)                  # scalar per bin
    - bias_by_region(region_name)                  # scalar per region
    - bias_by_rainfall_magnitude(obs_bin)          # scalar per bin

Conventions:
    bias = forecast - observed   (positive = wet bias in AIFS)
    units: mm/day
    All means use ``skipna=True``; land-only mask applied for the
    stratifications (IMERG ocean quality differs from land).
"""

from __future__ import annotations

from .. import config


def mean_bias_map(ds):
    """Mean (forecast - observed) over the time dimension."""
    raise NotImplementedError


def rmse_map(ds):
    """sqrt(mean((forecast - observed)^2)) over time."""
    raise NotImplementedError


def bias_by_bsiso_phase(ds, bsiso_df):
    """Group dates by BSISO phase (0..8) and compute mean bias per phase."""
    raise NotImplementedError


def bias_by_elevation(ds, elevation_da):
    """Stratify pointwise bias by elevation bin."""
    raise NotImplementedError


def bias_by_region(ds):
    """Mean bias and RMSE averaged within each named region in
    :data:`config.REGIONS`.
    """
    raise NotImplementedError


def bias_by_rainfall_magnitude(ds):
    """Mean bias stratified by observed-rainfall bin."""
    raise NotImplementedError
