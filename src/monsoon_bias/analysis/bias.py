"""Bias diagnostics.

All inputs come from the master Zarr store (see
:mod:`monsoon_bias.processing.store`). Bias is defined as
``forecast − observed`` (positive = wet bias in the forecast); units are
``mm/day``. Reductions use ``skipna=True`` so partially-populated stores
(e.g., mid-batch) yield results over the days that are populated.

The forecast/observed pair defaults to ``aifs`` / ``imerg`` but every
function accepts ``forecast`` / ``observed`` overrides so the same code
can compute the ERA5-vs-IMERG baseline or the AIFS-vs-ERA5 residual.

Diagnostics produced (all on the common 0.25° grid unless noted):
    mean_bias_map(lat, lon)
    rmse_map(lat, lon)
    bias_by_region              -> bias/rmse/count per named region
    bias_by_elevation           -> bias/count per elevation bin
    bias_by_rainfall_magnitude  -> bias/count per observed-rainfall bin
    bias_by_bsiso_phase         -> deferred (see STATUS.md)

Conventions:
    bias = forecast - observed
    units: mm/day
    no land mask applied here — leave that to the plotting layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .. import config

_DEFAULT_FORECAST = "aifs"
_DEFAULT_OBSERVED = "imerg"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bias(ds: xr.Dataset, forecast: str, observed: str) -> xr.DataArray:
    if forecast not in ds.data_vars:
        raise KeyError(f"forecast var {forecast!r} not in store; have {list(ds.data_vars)}")
    if observed not in ds.data_vars:
        raise KeyError(f"observed var {observed!r} not in store; have {list(ds.data_vars)}")
    b = ds[forecast] - ds[observed]
    b.attrs["units"] = "mm/day"
    b.attrs["long_name"] = f"bias ({forecast} − {observed})"
    return b


# ---------------------------------------------------------------------------
# Maps over (lat, lon)
# ---------------------------------------------------------------------------

def mean_bias_map(
    ds: xr.Dataset,
    *,
    forecast: str = _DEFAULT_FORECAST,
    observed: str = _DEFAULT_OBSERVED,
) -> xr.DataArray:
    """Mean (forecast − observed) over the time dimension."""
    b = _bias(ds, forecast, observed)
    out = b.mean(dim="time", skipna=True)
    out.attrs.update(b.attrs)
    out.attrs["long_name"] = f"mean bias ({forecast} − {observed})"
    out.attrs["n_days"] = int(b.notnull().any(dim=("lat", "lon")).sum())
    return out


def rmse_map(
    ds: xr.Dataset,
    *,
    forecast: str = _DEFAULT_FORECAST,
    observed: str = _DEFAULT_OBSERVED,
) -> xr.DataArray:
    """sqrt(mean((forecast − observed)^2)) over time."""
    b = _bias(ds, forecast, observed)
    out = np.sqrt((b ** 2).mean(dim="time", skipna=True))
    out.attrs["units"] = "mm/day"
    out.attrs["long_name"] = f"RMSE ({forecast} vs {observed})"
    out.attrs["n_days"] = int(b.notnull().any(dim=("lat", "lon")).sum())
    return out


# ---------------------------------------------------------------------------
# Stratifications -> 1D Datasets
# ---------------------------------------------------------------------------

def bias_by_region(
    ds: xr.Dataset,
    *,
    forecast: str = _DEFAULT_FORECAST,
    observed: str = _DEFAULT_OBSERVED,
) -> xr.Dataset:
    """Mean bias and RMSE averaged within each rectangular region in
    :data:`config.REGIONS`. Returns a Dataset indexed by region name.
    """
    b = _bias(ds, forecast, observed)
    names: list[str] = []
    biases: list[float] = []
    rmses: list[float] = []
    counts: list[int] = []
    for r in config.REGIONS:
        sub = b.sel(
            lat=slice(r.lat_min, r.lat_max),
            lon=slice(r.lon_min, r.lon_max),
        )
        if sub.size == 0:
            raise ValueError(f"region {r.name}: empty selection — check bbox vs store grid")
        names.append(r.name)
        biases.append(float(sub.mean(skipna=True)))
        rmses.append(float(np.sqrt((sub ** 2).mean(skipna=True))))
        counts.append(int(sub.notnull().sum()))
    out = xr.Dataset(
        data_vars={
            "bias": ("region", np.array(biases, dtype="float32")),
            "rmse": ("region", np.array(rmses, dtype="float32")),
            "count": ("region", np.array(counts, dtype="int64")),
        },
        coords={"region": np.array(names)},
        attrs={"forecast": forecast, "observed": observed, "units": "mm/day"},
    )
    return out


def bias_by_elevation(
    ds: xr.Dataset,
    elevation: xr.DataArray,
    *,
    forecast: str = _DEFAULT_FORECAST,
    observed: str = _DEFAULT_OBSERVED,
    bins: tuple[float, ...] = config.ELEVATION_BINS_M,
    labels: tuple[str, ...] = config.ELEVATION_LABELS,
) -> xr.Dataset:
    """Pointwise bias stratified by elevation bin.

    ``elevation`` is a 2D DataArray on the common grid (lat, lon) in
    meters. Bin edges are :data:`config.ELEVATION_BINS_M`; labels are
    :data:`config.ELEVATION_LABELS`.
    """
    if elevation.dims != ("lat", "lon"):
        raise ValueError(f"elevation must have dims (lat, lon), got {elevation.dims}")
    if len(labels) != len(bins) - 1:
        raise ValueError(f"len(labels)={len(labels)} != len(bins)-1={len(bins) - 1}")

    b = _bias(ds, forecast, observed)
    # Broadcast elevation over time for vectorized stratification.
    elev_b = elevation.broadcast_like(b)
    biases: list[float] = []
    counts: list[int] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (elev_b >= lo) & (elev_b < hi)
        sel = b.where(mask)
        biases.append(float(sel.mean(skipna=True)))
        counts.append(int(sel.notnull().sum()))
    out = xr.Dataset(
        data_vars={
            "bias": ("elev_bin", np.array(biases, dtype="float32")),
            "count": ("elev_bin", np.array(counts, dtype="int64")),
        },
        coords={
            "elev_bin": np.array(labels),
            "elev_bin_lo_m": ("elev_bin", np.array(bins[:-1], dtype="float32")),
            "elev_bin_hi_m": ("elev_bin", np.array(bins[1:], dtype="float32")),
        },
        attrs={"forecast": forecast, "observed": observed, "units": "mm/day"},
    )
    return out


def bias_by_rainfall_magnitude(
    ds: xr.Dataset,
    *,
    forecast: str = _DEFAULT_FORECAST,
    observed: str = _DEFAULT_OBSERVED,
    bins: tuple[float, ...] = config.RAINFALL_BINS_MM,
    labels: tuple[str, ...] = config.RAINFALL_LABELS,
) -> xr.Dataset:
    """Mean bias stratified by *observed* rainfall magnitude.

    Each (day, lat, lon) cell is binned by its observed (IMERG) value
    and the bias is averaged within each bin. Tests whether AIFS is
    biased high on light-rain days vs low on heavy-rain days — a common
    pattern in physics-light models.
    """
    if len(labels) != len(bins) - 1:
        raise ValueError(f"len(labels)={len(labels)} != len(bins)-1={len(bins) - 1}")

    b = _bias(ds, forecast, observed)
    obs = ds[observed]
    biases: list[float] = []
    counts: list[int] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (obs >= lo) & (obs < hi)
        sel = b.where(mask)
        biases.append(float(sel.mean(skipna=True)))
        counts.append(int(sel.notnull().sum()))
    out = xr.Dataset(
        data_vars={
            "bias": ("rain_bin", np.array(biases, dtype="float32")),
            "count": ("rain_bin", np.array(counts, dtype="int64")),
        },
        coords={
            "rain_bin": np.array(labels),
            "rain_bin_lo_mm": ("rain_bin", np.array(bins[:-1], dtype="float32")),
            "rain_bin_hi_mm": ("rain_bin", np.array(bins[1:], dtype="float32")),
        },
        attrs={"forecast": forecast, "observed": observed, "units": "mm/day"},
    )
    return out


# ---------------------------------------------------------------------------
# Deferred
# ---------------------------------------------------------------------------

def bias_by_bsiso_phase(ds: xr.Dataset, bsiso_df: pd.DataFrame) -> xr.Dataset:
    """Group days by BSISO phase (1..8) and compute mean bias per phase.

    Deferred until a BSISO index source is wired up. See
    :mod:`monsoon_bias.data.bsiso` for the recipe.
    """
    raise NotImplementedError(
        "BSISO stratification is deferred — no live index source. "
        "See STATUS.md → Known limitations."
    )
