"""Zarr master store I/O.

One Zarr store at :data:`config.ZARR_STORE` holds the full 122-day
monsoon season on the common 0.25° grid:

    Coords:
        time (122,)        IMD verifying dates, date-only
        lat  (129,)        India bbox @ 0.25°
        lon  (121,)
    Variables (all float32, dims (time, lat, lon), units mm/day):
        imerg              IMERG Final V07 IMD-day total
        era5               ERA5 reanalysis IMD-day total (baseline)
        aifs               AIFS forecast IMD-day total (3-day lead)

Design notes
------------
* **Pre-allocated NaN.** We know all 122 dates upfront from
  :func:`config.season_dates`, so :func:`init_store` writes a full
  (122, lat, lon) array of NaN for each variable. Per-day writes use
  ``xarray.Dataset.to_zarr(region={"time": slice(i, i+1)})``, which only
  rewrites one chunk on disk.
* **Chunking is (1, lat, lon).** One chunk per day matches the access
  pattern: every analysis is "load all days, then reduce/group along
  time," and per-day writes touch exactly one chunk.
* **Compression** is the zarr default (blosc). NaN-filled chunks
  compress to a few kB; the empty store is well under 1 MB.
* **Consolidated metadata** is written by :func:`consolidate` (call once
  after all days are written). Day-writes are unconsolidated for speed.
* **Bias is not stored.** It's a cheap derived quantity
  (``aifs - imerg`` or ``era5 - imerg``); the analysis computes it on
  the fly from the stored fields.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from .. import config

# Variables expected by the store. Keep these in sync with the analysis
# code; adding a new field requires re-initializing the store.
_VARIABLE_SPECS = {
    "imerg": {
        "long_name": "Observed rainfall (IMERG Final V07, IMD day total)",
        "source": "NASA GPM IMERG Final V07, half-hourly accumulated to IMD day",
    },
    "era5": {
        "long_name": "ERA5 reanalysis rainfall (IMD day total)",
        "source": "ECMWF ERA5 hourly tp, accumulated to IMD day",
    },
    "aifs": {
        "long_name": "AIFS forecast rainfall (3-day lead, IMD day total)",
        "source": "ECMWF AIFS-Single 1.1.0 via Earth2Studio, 03 UTC init, leads 78-96h",
    },
}


# ---------------------------------------------------------------------------
# Create / open
# ---------------------------------------------------------------------------

def init_store(path: Path | None = None, *, overwrite: bool = False) -> Path:
    """Create the master Zarr store, pre-allocated with NaN for every day
    in the configured monsoon season.

    Raises ``FileExistsError`` if the store already exists and
    ``overwrite`` is False.
    """
    if path is None:
        path = config.ZARR_STORE
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Zarr store already exists: {path}. Pass overwrite=True to replace.")

    dates = config.season_dates()                # 122 dates
    grid = config.common_grid()
    lat, lon = grid["lat"], grid["lon"]
    shape = (len(dates), len(lat), len(lon))

    data_vars = {
        name: (
            ("time", "lat", "lon"),
            np.full(shape, np.nan, dtype="float32"),
            {"units": "mm/day", **specs},
        )
        for name, specs in _VARIABLE_SPECS.items()
    }
    ds = xr.Dataset(
        coords={"time": dates, "lat": lat, "lon": lon},
        data_vars=data_vars,
        attrs={
            "title": "Monsoon bias prototype master store",
            "season": f"{config.SEASON_START.date()} → {config.SEASON_END.date()}",
            "grid_resolution_deg": config.GRID_RESOLUTION,
            "bbox_lat_min_max_lon_min_max": list(config.INDIA_BBOX),
            "imd_day_start_utc": config.IMD_DAY_START_HOUR_UTC,
        },
    )
    encoding = {name: {"chunks": (1, len(lat), len(lon))} for name in _VARIABLE_SPECS}
    mode = "w" if overwrite else "w-"
    ds.to_zarr(path, mode=mode, encoding=encoding, consolidated=False)
    return path


def open_store(path: Path | None = None) -> xr.Dataset:
    """Open the master store as an xarray.Dataset (read-only-ish)."""
    if path is None:
        path = config.ZARR_STORE
    # Try consolidated metadata first (fast); fall back if not consolidated yet.
    try:
        return xr.open_zarr(path, consolidated=True)
    except (KeyError, ValueError):
        return xr.open_zarr(path, consolidated=False)


def consolidate(path: Path | None = None) -> None:
    """Consolidate Zarr metadata. Call once after the batch of day-writes
    finishes — speeds up subsequent reads.
    """
    if path is None:
        path = config.ZARR_STORE
    zarr.consolidate_metadata(str(path))


# ---------------------------------------------------------------------------
# Per-day writes
# ---------------------------------------------------------------------------

def _validate_field(name: str, da: xr.DataArray, store: xr.Dataset) -> None:
    if str(da.attrs.get("units", "")).lower() not in ("mm/day", "mm d-1"):
        raise ValueError(f"{name}: units must be mm/day, got {da.attrs.get('units')!r}")
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"{name}: needs lat and lon dims, got {da.dims}")
    if da.shape != (store.sizes["lat"], store.sizes["lon"]):
        raise ValueError(
            f"{name}: shape {da.shape} != store grid {(store.sizes['lat'], store.sizes['lon'])}"
        )
    for coord in ("lat", "lon"):
        if not np.allclose(da[coord].values, store[coord].values, atol=1e-6):
            raise ValueError(f"{name}: {coord} values do not match store grid.")


def write_day(
    verifying_date: pd.Timestamp,
    *,
    imerg: xr.DataArray | None = None,
    era5: xr.DataArray | None = None,
    aifs: xr.DataArray | None = None,
    path: Path | None = None,
) -> None:
    """Write one IMD day's fields into the store.

    Any of ``imerg`` / ``era5`` / ``aifs`` may be ``None`` (e.g., AIFS not
    yet computed); only the provided fields are written. Each input must
    be a 2D DataArray on the common grid with units ``mm/day``.
    """
    if path is None:
        path = config.ZARR_STORE
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Store does not exist: {path}. Call init_store first.")

    fields = {k: v for k, v in {"imerg": imerg, "era5": era5, "aifs": aifs}.items() if v is not None}
    if not fields:
        raise ValueError("write_day called with all fields None.")

    store = open_store(path)
    verifying_date = pd.Timestamp(verifying_date).normalize()
    if verifying_date not in pd.DatetimeIndex(store.time.values):
        raise ValueError(
            f"{verifying_date.date()} not in store time index "
            f"({config.SEASON_START.date()}..{config.SEASON_END.date()})."
        )
    t_idx = int(np.where(store.time.values == np.datetime64(verifying_date))[0][0])
    grid_lat = store.lat.values
    grid_lon = store.lon.values
    store.close()

    # Build a single-time Dataset matching the store schema for the fields
    # being written, then region-write.
    write_data = {}
    for name, da in fields.items():
        _validate_field(name, da, xr.Dataset(coords={"lat": grid_lat, "lon": grid_lon}))
        da3d = da.astype("float32").expand_dims(time=[verifying_date])
        # Make sure coord values are exact (validate already confirmed within tol).
        da3d = da3d.assign_coords(lat=grid_lat, lon=grid_lon)
        write_data[name] = da3d

    ds = xr.Dataset(write_data)
    # to_zarr(region=...) requires that every coord in the dataset spans the
    # region's dimension. lat/lon (and any spurious coords like xarray-regrid's
    # 'number') don't — and they're already on disk from init_store, so we just
    # drop them from the in-memory dataset before writing.
    drop = [c for c in ds.coords if c != "time"]
    if drop:
        ds = ds.drop_vars(drop)
    ds.to_zarr(path, region={"time": slice(t_idx, t_idx + 1)}, consolidated=False)
