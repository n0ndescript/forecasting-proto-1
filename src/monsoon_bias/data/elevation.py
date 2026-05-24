"""Elevation data (NOAA ETOPO 2022, 60-arc-second surface) for the
orographic bias stratification.

Source
------
NCEI ETOPO 2022, "surface" variant (topographic surface — ice surface
where ice exists). 60-arc-second (~1.8 km) resolution is plenty given
we average down to 0.25° (~28 km) anyway.

Two endpoints:
* OPENDAP (preferred — crop the India region without downloading the
  full 933 MB global file): netCDF4 reads it transparently via DAP.
* HTTPS bulk download (fallback): the full global file is ~933 MB.

Variable name in the file is ``z`` (Float32). Negative values are
bathymetry; we clip those to 0 since the analysis is land-focused.

The cached India crop lives at :data:`config.ELEVATION_DIR` /
``etopo2022_60s_india.nc`` (~13 MB, one-time download).
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr
import xarray_regrid  # noqa: F401 — registers `.regrid` accessor

from .. import config


ETOPO_OPENDAP_URL = (
    "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/"
    "60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
)
ETOPO_HTTPS_URL = (
    "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/"
    "60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
)
# Cropped India cache location.
_INDIA_CROP_NAME = "etopo2022_60s_india.nc"


def download_etopo_india(output_path: Path | None = None,
                          *, pad_deg: float = 0.5) -> Path:
    """Download a cropped India subset of ETOPO 2022 60-arc-sec.

    Padding ensures conservative regridding has full source cells at the
    edges of our 0.25° target grid.

    Returns the local NetCDF path. Idempotent — skips if cache exists.
    """
    if output_path is None:
        output_path = config.ELEVATION_DIR / _INDIA_CROP_NAME
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lat_min, lat_max, lon_min, lon_max = config.INDIA_BBOX
    # Open via OPENDAP — netcdf4 backend follows DAP URLs transparently.
    ds = xr.open_dataset(ETOPO_OPENDAP_URL)
    crop = ds.sel(
        lat=slice(lat_min - pad_deg, lat_max + pad_deg),
        lon=slice(lon_min - pad_deg, lon_max + pad_deg),
    )
    crop.load().to_netcdf(output_path)
    ds.close()
    return output_path


def load_elevation_on_common_grid(path: Path | None = None) -> xr.DataArray:
    """Load ETOPO India and conservatively regrid to the common 0.25° grid.

    Returns a DataArray with dims (lat, lon), units ``m``, name
    ``elevation``. Ocean (negative ETOPO) is clipped to 0.
    """
    if path is None:
        path = config.ELEVATION_DIR / _INDIA_CROP_NAME
    if not path.exists():
        download_etopo_india(path)

    ds = xr.open_dataset(path)
    var = next((v for v in ("z", "elevation", "topo", "Band1") if v in ds), None)
    if var is None:
        raise ValueError(f"No elevation variable in {path}; have {list(ds.data_vars)}")
    da = ds[var]

    # xarray-regrid wants ascending coords.
    for coord in ("lat", "lon"):
        if da[coord].values[0] > da[coord].values[-1]:
            da = da.isel({coord: slice(None, None, -1)})

    # Mask ocean (negative bathymetry) to 0 — analysis is land-focused.
    da = da.where(da > 0, 0)

    grid = config.common_grid()
    target = xr.Dataset(coords={"lat": grid["lat"], "lon": grid["lon"]})
    elev = da.regrid.conservative(target, latitude_coord="lat")
    elev.name = "elevation"
    elev.attrs = {
        "units": "m",
        "long_name": "Surface elevation (mean within 0.25° cell, ocean clipped to 0)",
        "source": "NOAA NCEI ETOPO 2022, 60-arc-second surface variant",
        "regrid_method": "conservative (area-weighted mean)",
    }
    return elev
