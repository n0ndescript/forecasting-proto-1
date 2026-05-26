"""Accumulate sub-daily rainfall to IMD-aligned 24-hour totals.

IMD's "daily rainfall" runs 08:30 IST → 08:30 IST (next day), i.e.
03:00 UTC → 03:00 UTC. Every accumulation in this project respects that
window. The end timestamp is exclusive.

Unit conventions after accumulation: **mm per day** (not mm/s, not
kg/m²). Functions here assert units to avoid silent unit-mismatch bugs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .. import config

# Number of half-hourly IMERG timestamps in one IMD day (24 hr × 2/hr).
_HHR_PER_DAY = 48
# AIFS native step in hours.
_AIFS_STEP_HOURS = 6


def accumulate_imerg_half_hourly_to_imd_day(
    paths: list[Path],
    verifying_date: pd.Timestamp,
    bbox: tuple[float, float, float, float] | None = None,
) -> xr.DataArray:
    """Sum 48 IMERG half-hourly granules to one mm/day field for the
    IMD day ending on ``verifying_date``.

    IMERG ``precipitation`` units are ``mm/hr`` (the half-hour mean
    rate). A single granule therefore contributes ``rate * 0.5 hr`` mm.
    Summing the 48 half-hour contributions in the window gives the IMD-
    day total in mm.

    Returns a DataArray with dims (lat, lon), units ``mm/day``, cropped
    to ``bbox`` (defaults to :data:`config.INDIA_BBOX`).
    """
    if bbox is None:
        bbox = config.INDIA_BBOX
    lat_min, lat_max, lon_min, lon_max = bbox

    start_utc, end_utc = config.imd_day_window_utc(verifying_date)

    rates: list[xr.DataArray] = []
    times: list[pd.Timestamp] = []
    for p in sorted(paths):
        # IMERG HDF5 uses a Julian-like calendar so xarray decodes times as
        # cftime by default; pandas can't compare those to Timestamp. Convert
        # via the ISO string round-trip.
        ds = xr.open_dataset(p, group="Grid")
        if "precipitation" not in ds:
            raise ValueError(f"{p.name}: no 'precipitation' variable.")
        # IMERG dims are (time=1, lon, lat); reorder & crop.
        da = ds["precipitation"].squeeze("time", drop=False)
        if "lon" in da.dims and "lat" in da.dims:
            da = da.transpose("lat", "lon")
        else:
            raise ValueError(f"{p.name}: unexpected dims {da.dims}")
        da = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

        units = str(ds["precipitation"].attrs.get("units", "")).lower()
        if units not in ("mm/hr", "mm h-1", "mm hr-1"):
            raise ValueError(f"{p.name}: unexpected precip units {units!r} (expected mm/hr).")

        raw_t = ds["time"].values.item()
        t = pd.Timestamp(raw_t.isoformat() if hasattr(raw_t, "isoformat") else raw_t)
        if start_utc <= t < end_utc:
            rates.append(da.load())
            times.append(t)
        ds.close()

    if len(rates) != _HHR_PER_DAY:
        raise ValueError(
            f"Expected {_HHR_PER_DAY} half-hourly granules in window, got {len(rates)} "
            f"for IMD day {verifying_date.date()} (window {start_utc} → {end_utc})."
        )

    stacked = xr.concat(rates, dim=pd.Index(times, name="time"))
    # Each half-hour contributes rate (mm/hr) × 0.5 hr → mm.
    total_mm = (stacked * 0.5).sum(dim="time", skipna=False)
    # Replace IMERG fill values (~-9999.9) with NaN if any survived.
    total_mm = total_mm.where(total_mm > -1.0)

    total_mm.name = "rainfall_mm_per_day"
    total_mm.attrs = {
        "units": "mm/day",
        "long_name": "IMD-day accumulated rainfall",
        "source": "GPM IMERG Final V07, half-hourly",
        "accumulation_window_utc": f"{start_utc.isoformat()} → {end_utc.isoformat()} (exclusive end)",
        "verifying_date_imd": verifying_date.strftime("%Y-%m-%d"),
        "n_granules": len(rates),
    }
    return total_mm


def accumulate_era5_hourly_to_imd_day(
    ds: xr.Dataset,
    verifying_date: pd.Timestamp,
    var: str = "tp",
) -> xr.DataArray:
    """Sum hourly ERA5 ``tp`` over the IMD-day window.

    ERA5 ``tp`` is in **meters of water-equivalent accumulated over the
    preceding hour**. Convert to mm via × 1000, then sum the 24 hourly
    values whose stamps fall in (start, start+24h].
    """
    start_utc, end_utc = config.imd_day_window_utc(verifying_date)
    if var not in ds:
        raise ValueError(f"ERA5 dataset missing variable {var!r}; have {list(ds.data_vars)}")
    da = ds[var]
    units = str(da.attrs.get("units", "")).lower()
    if units not in ("m", "m of water equivalent", "metres", "meter"):
        raise ValueError(f"ERA5 {var} units {units!r}; expected meters.")
    # ERA5 hourly tp at time T is the accumulation over (T-1h, T].
    # So the 24 stamps strictly greater than start_utc and ≤ end_utc
    # cover the IMD window exactly.
    sel = da.sel(time=slice(start_utc + pd.Timedelta(seconds=1), end_utc))
    if sel.sizes.get("time", 0) != 24:
        raise ValueError(
            f"Expected 24 ERA5 hourly timestamps in window, got {sel.sizes.get('time', 0)} "
            f"for IMD day {verifying_date.date()}."
        )
    total_mm = (sel * 1000.0).sum(dim="time", skipna=False)
    total_mm.name = "rainfall_mm_per_day"
    total_mm.attrs = {
        "units": "mm/day",
        "long_name": "IMD-day accumulated rainfall",
        "source": "ERA5 reanalysis, hourly tp",
        "accumulation_window_utc": f"{start_utc.isoformat()} → {end_utc.isoformat()}",
        "verifying_date_imd": verifying_date.strftime("%Y-%m-%d"),
    }
    return total_mm


def accumulate_aifs_to_imd_day(
    ds: xr.Dataset,
    verifying_date: pd.Timestamp,
    init_time: pd.Timestamp,
    var: str | None = None,
) -> xr.DataArray:
    """Sum 4 AIFS 6-h precipitation steps to one IMD-day total.

    AIFS native step is 6 h and ``tp06`` is the per-step accumulation.
    For step times to align with the IMD window (03→03 UTC) we **must**
    initialize AIFS at 03 UTC: then valid times land on 09, 15, 21, 03
    UTC, and the four steps with valid times {V+09h, V+15h, V+21h,
    V+1d+03h} accumulate exactly the IMD-day total (where V = the
    verifying date at 00 UTC).

    Concretely, for a verifying date 2025-07-15:
        init_time     = 2025-07-12 03:00 UTC  (lead = 3 days)
        wanted leads  = 78h, 84h, 90h, 96h  → valid 07-15 09/15/21, 07-16 03
        sum of tp06   = total mm/day for IMD day 2025-07-15

    The forecast must therefore extend at least 96 h (nsteps ≥ 16 in
    Earth2Studio terms). See :mod:`monsoon_bias.forecast.run_aifs`.

    Defensive clipping
    ------------------
    AIFS 1.0.0 (the original release) sometimes produced negative
    precipitation values; the issue was fixed in 1.1.0 (released
    2025-08-27). Earth2Studio 0.14.0 ships 1.1.0 by default, but if the
    cached package is older we'd silently get unphysical totals. We
    clip negatives to zero and emit a warning if more than 0.1 % of
    cells in the IMD window are negative — that's the signature of a
    stale model package.

    Args:
        ds: AIFS output dataset; must contain ``tp06`` (or ``tp``)
            indexed by a time-like coord.
        verifying_date: IMD calendar date (00 UTC normalized).
        init_time: AIFS initialization time, must be 03:00 UTC.
        var: precip variable name. If None, tries ``tp06`` then ``tp``.

    Returns:
        DataArray with dims (lat, lon), units ``mm/day``.
    """
    import warnings

    if init_time.hour != 3 or init_time.minute != 0 or init_time.second != 0:
        raise ValueError(
            f"AIFS init time must be 03:00 UTC for IMD alignment; got {init_time}."
        )

    start_utc, end_utc = config.imd_day_window_utc(verifying_date)

    # Resolve the precip variable. tp06 is what Earth2Studio 0.14.0
    # surfaces for AIFS; older serializations may use tp.
    if var is None:
        var = next((v for v in ("tp06", "tp") if v in ds), None)
        if var is None:
            raise ValueError(
                f"AIFS dataset has no tp06 or tp variable; have {list(ds.data_vars)}."
            )
    elif var not in ds:
        raise ValueError(f"AIFS dataset missing {var!r}; have {list(ds.data_vars)}.")
    da = ds[var]

    # AIFS tp06 native units: meters of water equivalent (ECMWF convention).
    # Earth2Studio 0.14.0 emits `tp06` with NO units attribute set; we
    # treat that as the documented ECMWF default (meters). Any explicit
    # units string overrides.
    units = str(da.attrs.get("units", "")).strip().lower()
    if units in ("", "m", "meter", "metre", "m of water equivalent"):
        scale = 1000.0
    elif units in ("mm", "kg m-2", "kg/m^2"):
        scale = 1.0
    else:
        raise ValueError(f"AIFS {var} units {units!r}; expected m or mm.")

    # Earth2Studio 0.14 emits AIFS output with two time-like coords:
    # ``time`` (size 1, the init time) and ``lead_time`` (timedeltas
    # from init: 0, 6h, 12h, ..., 96h). The 4 IMD-day windows correspond
    # to leads 78, 84, 90, 96 h (since init = 03 UTC, lead = 3 days).
    # Older serializations stored absolute valid-time stamps in a single
    # ``time`` or ``valid_time`` dim — keep that path for compatibility.
    if "lead_time" in da.dims:
        if "time" in da.dims and da.sizes.get("time", 0) == 1:
            ds_init = pd.Timestamp(da["time"].values.item())
            if ds_init != init_time:
                raise ValueError(
                    f"AIFS dataset init time {ds_init} does not match "
                    f"argument init_time {init_time}."
                )
            da = da.squeeze("time", drop=True)
        wanted_leads = [pd.Timedelta(hours=h) for h in (78, 84, 90, 96)]
        present_leads = pd.to_timedelta(da["lead_time"].values)
        missing = [t for t in wanted_leads if t not in list(present_leads)]
        if missing:
            raise ValueError(
                f"AIFS output missing lead_times {missing} required for IMD day "
                f"{verifying_date.date()}. Available: {list(present_leads)}"
            )
        selected = (da.sel(lead_time=wanted_leads) * scale)
        time_dim = "lead_time"
    else:
        time_dim = "time" if "time" in da.dims else (
            "valid_time" if "valid_time" in da.dims else None)
        if time_dim is None:
            raise ValueError(f"AIFS {var} has no time-like dim; dims={da.dims}.")

        wanted = [
            verifying_date.normalize() + pd.Timedelta(hours=9),
            verifying_date.normalize() + pd.Timedelta(hours=15),
            verifying_date.normalize() + pd.Timedelta(hours=21),
            verifying_date.normalize() + pd.Timedelta(days=1, hours=3),
        ]
        present = pd.to_datetime(da[time_dim].values)
        missing = [t for t in wanted if t not in present]
        if missing:
            raise ValueError(
                f"AIFS output missing steps {missing} required for IMD day "
                f"{verifying_date.date()}. Available: {list(present)}"
            )

        selected = (da.sel({time_dim: wanted}) * scale)
    # Clip negatives; warn if many — that indicates AIFS 1.0.0 (pre-fix).
    n_neg = int((selected < 0).sum())
    n_total = int(selected.size)
    if n_neg / max(n_total, 1) > 1e-3:
        warnings.warn(
            f"AIFS {var}: {n_neg}/{n_total} ({100 * n_neg/n_total:.2f}%) cells "
            "negative. This is the signature of AIFS 1.0.0 (pre-2025-08-27); "
            "upgrade to 1.1.0 by clearing the Earth2Studio package cache."
        )
    selected = selected.where(selected >= 0, 0.0)

    total_mm = selected.sum(dim=time_dim, skipna=False)
    total_mm.name = "rainfall_mm_per_day"
    total_mm.attrs = {
        "units": "mm/day",
        "long_name": "IMD-day accumulated rainfall (AIFS forecast)",
        "source": f"AIFS forecast, init {init_time.isoformat()}, leads 78–96h",
        "accumulation_window_utc": f"{start_utc.isoformat()} → {end_utc.isoformat()}",
        "verifying_date_imd": verifying_date.strftime("%Y-%m-%d"),
        "negative_cells_clipped": n_neg,
    }
    return total_mm
