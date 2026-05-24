"""Regridding to the project's common 0.25° grid.

Both source and target grids are rectilinear lat/lon, so we use
``xarray-regrid`` (pure Python) rather than xESMF/ESMF. The
``conservative`` method preserves area-integrated quantities (essential
for precipitation; bilinear is wrong and silently loses water).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

import xarray_regrid  # noqa: F401 — registers ``.regrid`` accessor

from .. import config


def _common_target() -> xr.Dataset:
    """Return an empty xarray.Dataset with just the target grid coords."""
    grid = config.common_grid()
    return xr.Dataset(coords={"lat": grid["lat"], "lon": grid["lon"]})


def _ensure_ascending(da: xr.DataArray) -> xr.DataArray:
    """xarray-regrid requires monotonically increasing coords. IMERG and
    some ERA5 outputs are stored north→south; flip if needed.
    """
    for coord in ("lat", "lon"):
        if coord in da.coords and da[coord].size > 1:
            vals = da[coord].values
            if vals[0] > vals[-1]:
                da = da.isel({coord: slice(None, None, -1)})
    return da


def regrid_precip(da: xr.DataArray, target: xr.Dataset | None = None) -> xr.DataArray:
    """Conservatively regrid a precipitation field to the common 0.25° grid.

    Refuses anything but conservative. Validates units are ``mm/day``
    before regridding so we don't accidentally regrid mm/s or m and
    silently corrupt totals.
    """
    if str(da.attrs.get("units", "")).lower() not in ("mm/day", "mm d-1"):
        raise ValueError(
            f"regrid_precip requires units 'mm/day' (got {da.attrs.get('units')!r}). "
            "Accumulate to a daily total first."
        )
    if target is None:
        target = _common_target()
    da = _ensure_ascending(da)
    out = da.regrid.conservative(target, latitude_coord="lat")
    # Preserve metadata that xarray-regrid drops.
    out.attrs = dict(da.attrs)
    out.attrs["regrid_method"] = "conservative"
    out.attrs["regrid_target"] = (
        f"India bbox {config.INDIA_BBOX} @ {config.GRID_RESOLUTION}°"
    )
    return out


def regrid_continuous(da: xr.DataArray, target: xr.Dataset | None = None,
                      method: str = "linear") -> xr.DataArray:
    """Regrid a continuous (non-conserved) field — e.g., 2 m temperature.

    Default is bilinear (``linear``). NEVER use this for precipitation.
    """
    if method not in ("linear", "nearest", "cubic"):
        raise ValueError(f"method must be linear/nearest/cubic, got {method!r}.")
    if target is None:
        target = _common_target()
    da = _ensure_ascending(da)
    fn = getattr(da.regrid, method)
    out = fn(target)
    out.attrs = dict(da.attrs)
    out.attrs["regrid_method"] = method
    return out


def verify_coastline_alignment(da_a: xr.DataArray, da_b: xr.DataArray,
                               threshold: float = 1e-6) -> None:
    """Assert that two regridded fields share identical lat/lon arrays.

    Cheap sanity check before computing bias = a − b: silent coord
    mismatch is the most common source of garbage diagnostics.
    """
    for coord in ("lat", "lon"):
        a = da_a[coord].values
        b = da_b[coord].values
        if a.shape != b.shape:
            raise ValueError(f"{coord} shape mismatch: {a.shape} vs {b.shape}.")
        if np.max(np.abs(a - b)) > threshold:
            raise ValueError(
                f"{coord} values differ by > {threshold}; max diff "
                f"{float(np.max(np.abs(a - b)))}."
            )
