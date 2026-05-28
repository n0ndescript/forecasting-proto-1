"""scripts/05_compute_bias.py

CPU. Reads the master Zarr store and writes bias diagnostics as
NetCDFs under :data:`config.OUTPUTS_DIR` / ``bias`` /:

    mean_bias_map_aifs.nc       AIFS−IMERG mean bias on the common grid
    rmse_map_aifs.nc            AIFS vs IMERG RMSE
    mean_bias_map_era5.nc       ERA5−IMERG (baseline)
    rmse_map_era5.nc            ERA5 vs IMERG RMSE
    mean_bias_map_residual.nc   AIFS−ERA5 (AIFS-specific residual)
    bias_by_region.nc           per-region bias/RMSE (Dataset)
    bias_by_elevation.nc        per-elevation-bin bias (Dataset)
    bias_by_rainfall_magnitude.nc  per-obs-mag-bin bias (Dataset)

Run:
    uv run python scripts/05_compute_bias.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from monsoon_bias import config  # noqa: E402
from monsoon_bias.analysis import bias as biaslib  # noqa: E402
from monsoon_bias.data import elevation as elev_lib  # noqa: E402
from monsoon_bias.processing import store  # noqa: E402

BIAS_OUT_DIR = config.OUTPUTS_DIR / "bias"


def main() -> int:
    print("=" * 72)
    print("Compute bias diagnostics")
    print(f"Zarr store : {config.ZARR_STORE}")
    print(f"Outputs    : {BIAS_OUT_DIR}")
    print("=" * 72)

    if not config.ZARR_STORE.exists():
        sys.exit(f"[fail] Zarr store missing: {config.ZARR_STORE}")
    BIAS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = store.open_store().load()
    n_aifs = int((~np.isnan(ds.aifs).all(dim=("lat", "lon"))).sum())
    n_imerg = int((~np.isnan(ds.imerg).all(dim=("lat", "lon"))).sum())
    n_era5 = int((~np.isnan(ds.era5).all(dim=("lat", "lon"))).sum())
    print(f"[main] populated: aifs={n_aifs}, imerg={n_imerg}, era5={n_era5} (of {len(ds.time)})")

    if n_aifs == 0 or n_imerg == 0:
        sys.exit("[fail] need at least one populated day for both aifs and imerg.")

    # ---- Maps ----
    print("[1/8] mean bias map  (AIFS − IMERG)")
    biaslib.mean_bias_map(ds).to_netcdf(BIAS_OUT_DIR / "mean_bias_map_aifs.nc")
    print("[2/8] RMSE map       (AIFS vs IMERG)")
    biaslib.rmse_map(ds).to_netcdf(BIAS_OUT_DIR / "rmse_map_aifs.nc")

    print("[3/8] mean bias map  (ERA5 − IMERG, baseline)")
    biaslib.mean_bias_map(ds, forecast="era5").to_netcdf(BIAS_OUT_DIR / "mean_bias_map_era5.nc")
    print("[4/8] RMSE map       (ERA5 vs IMERG)")
    biaslib.rmse_map(ds, forecast="era5").to_netcdf(BIAS_OUT_DIR / "rmse_map_era5.nc")

    print("[5/8] residual map   (AIFS − ERA5)")
    biaslib.mean_bias_map(ds, forecast="aifs", observed="era5").to_netcdf(
        BIAS_OUT_DIR / "mean_bias_map_residual.nc"
    )

    # ---- Stratifications ----
    print("[6/8] bias by region")
    biaslib.bias_by_region(ds).to_netcdf(BIAS_OUT_DIR / "bias_by_region.nc")

    print("[7/8] bias by elevation")
    elev = elev_lib.load_elevation_on_common_grid()
    biaslib.bias_by_elevation(ds, elev).to_netcdf(BIAS_OUT_DIR / "bias_by_elevation.nc")
    # Also persist the elevation field so script 06 doesn't re-download.
    elev.to_netcdf(BIAS_OUT_DIR / "_elevation_on_common_grid.nc")

    print("[8/8] bias by observed-rainfall magnitude")
    biaslib.bias_by_rainfall_magnitude(ds).to_netcdf(
        BIAS_OUT_DIR / "bias_by_rainfall_magnitude.nc"
    )

    print()
    print(f"[main] Wrote 8 diagnostics to {BIAS_OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
