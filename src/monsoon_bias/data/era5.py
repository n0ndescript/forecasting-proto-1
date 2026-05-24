"""ERA5 downloader via the Copernicus CDS API.

Two use cases:

1. **Baseline hourly precipitation** (:func:`download_era5_precip_imd_day`)
   for the ERA5-vs-IMERG comparison panel. This is the data we
   actually need to fetch ourselves.

2. **AIFS initial conditions** — Earth2Studio's :class:`CDS` data source
   fetches all 89 input fields automatically when the forecast runs, so
   we do not pre-download ICs here. See
   :mod:`monsoon_bias.forecast.run_aifs`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config


def _cds_area() -> list[float]:
    """CDS API expects area as [North, West, South, East]."""
    lat_min, lat_max, lon_min, lon_max = config.INDIA_BBOX
    return [lat_max, lon_min, lat_min, lon_max]


def _load_cds_credentials() -> tuple[str | None, str | None, Path | None]:
    """Find .cdsapirc in the project root or ~/. Returns (url, key, path)."""
    for path in (config.PROJECT_ROOT / ".cdsapirc", Path.home() / ".cdsapirc"):
        if not path.exists():
            continue
        url, key = None, None
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("url:"):
                url = line.split(":", 1)[1].strip()
            elif line.startswith("key:"):
                key = line.split(":", 1)[1].strip()
        if url and key:
            return url, key, path
    return None, None, None


def _cds_client():
    import cdsapi  # local import keeps non-CDS callers fast
    url, key, src = _load_cds_credentials()
    if not (url and key):
        raise RuntimeError("No CDS credentials found in .cdsapirc.")
    return cdsapi.Client(url=url, key=key)


def download_era5_precip_imd_day(
    verifying_date: pd.Timestamp,
    output_path: Path | None = None,
) -> Path:
    """Download hourly ERA5 total precipitation covering the IMD day
    ending on ``verifying_date``.

    We request 48 hourly stamps (full verifying day + next day) and let
    :func:`monsoon_bias.processing.accumulate.accumulate_era5_hourly_to_imd_day`
    slice the 24 stamps in the IMD window. Slightly oversize but
    simpler than splitting the request.

    Returns the NetCDF path. Skips download if file already exists.
    """
    if output_path is None:
        output_path = config.ERA5_DIR / f"era5_tp_{verifying_date.strftime('%Y-%m-%d')}.nc"
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    next_day = verifying_date + pd.Timedelta(days=1)
    years = sorted({verifying_date.year, next_day.year})
    months = sorted({verifying_date.month, next_day.month})
    days = sorted({verifying_date.day, next_day.day})

    request = {
        "product_type": ["reanalysis"],
        "variable": ["total_precipitation"],
        "year": [f"{y:04d}" for y in years],
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in days],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": _cds_area(),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    client = _cds_client()
    client.retrieve("reanalysis-era5-single-levels", request, str(output_path))
    return output_path
