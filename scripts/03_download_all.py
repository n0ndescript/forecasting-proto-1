"""scripts/03_download_all.py

Batch-download IMERG half-hourly + ERA5 hourly precipitation for the
full 2025 monsoon (122 days, Jun 1 – Sep 30), accumulate to mm/day,
regrid to the common 0.25° grid, and write to the Zarr master store.

This script is CPU-only (no GPU needed). The AIFS forecasts run
separately in scripts/04_run_forecasts.py on the GPU host.

Properties:
* **Resumable.** Days already populated in the Zarr store are skipped.
  Raw IMERG/ERA5 files already on disk are reused (the downloaders are
  idempotent — they don't re-fetch existing files).
* **Failure-tolerant.** A failed day is logged and the loop continues;
  failures are summarized at the end. Re-run to retry.
* **Background-friendly.** One log line per day, no progress bars.
  Safe to nohup / run_in_background.

Disk footprint at completion (~8.6 GB):
* IMERG half-hourly:  ~70 MB/day × 122  ≈ 8.5 GB
* ERA5 hourly tp:     ~1 MB/day  × 122  ≈ 0.13 GB
* Zarr master store:  ~3 MB total (compressed; 122 × 129 × 121 × 3 vars)

Run:
    uv run python scripts/03_download_all.py
"""

from __future__ import annotations

import concurrent.futures
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from monsoon_bias import config  # noqa: E402
from monsoon_bias.data import era5 as era5_dl, imerg as imerg_dl  # noqa: E402
from monsoon_bias.data._earthdata import load_edl_token, EarthdataError  # noqa: E402
from monsoon_bias.processing import accumulate, regrid, store  # noqa: E402


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def preflight() -> None:
    """Fail fast if credentials are missing or unusable. Avoids losing
    minutes/hours before discovering an auth problem."""
    # EDL token (IMERG)
    try:
        load_edl_token()
    except EarthdataError as exc:
        sys.exit(f"[preflight] EDL token: {exc}")
    # CDS credentials (ERA5)
    url, key, src = era5_dl._load_cds_credentials()
    if not (url and key):
        sys.exit("[preflight] No .cdsapirc found in project root or ~/. Cannot fetch ERA5.")
    print(f"[preflight] EDL token: present, CDS creds: present ({src})")


def ensure_store() -> None:
    """Create the Zarr store on first run; otherwise leave it alone."""
    if not config.ZARR_STORE.exists():
        print(f"[preflight] Creating Zarr store at {config.ZARR_STORE}")
        store.init_store()
    else:
        print(f"[preflight] Zarr store exists at {config.ZARR_STORE}")


def missing_dates() -> list[pd.Timestamp]:
    """Return the verifying dates that still need IMERG or ERA5 populated."""
    ds = store.open_store()
    times = pd.DatetimeIndex(ds.time.values)
    todo: list[pd.Timestamp] = []
    for t in times:
        imerg_null = bool(ds.imerg.sel(time=t).isnull().all())
        era5_null = bool(ds.era5.sel(time=t).isnull().all())
        if imerg_null or era5_null:
            todo.append(pd.Timestamp(t))
    ds.close()
    return todo


# ---------------------------------------------------------------------------
# Per-day pipeline
# ---------------------------------------------------------------------------

def _open_era5_for_accum(path: Path) -> xr.Dataset:
    """Open CDS ERA5 NetCDF and standardize coord names."""
    ds = xr.open_dataset(path)
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    return ds.rename(rename) if rename else ds


def process_one_day(v: pd.Timestamp) -> tuple[bool, str]:
    """Download → accumulate → regrid → store-write for one verifying date.

    IMERG (48 granules, parallel internally) and ERA5 (one CDS request)
    are fetched concurrently in a 2-thread pool — they're independent
    until the accumulate step. Each day's wall-clock is bounded by
    max(IMERG_time, ERA5_time) ≈ ~60 s.

    Returns (success, message).
    """
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            imerg_future = pool.submit(imerg_dl.download_imerg_for_imd_day, v)
            era5_future = pool.submit(era5_dl.download_era5_precip_imd_day, v)
            imerg_paths = imerg_future.result()
            era5_path = era5_future.result()

        imerg_native = accumulate.accumulate_imerg_half_hourly_to_imd_day(imerg_paths, v)
        imerg = regrid.regrid_precip(imerg_native)

        era5_ds = _open_era5_for_accum(era5_path)
        era5_native = accumulate.accumulate_era5_hourly_to_imd_day(era5_ds, v, var="tp")
        era5 = regrid.regrid_precip(era5_native)
        era5_ds.close()

        store.write_day(v, imerg=imerg, era5=era5)

        msg = (f"imerg mean={float(imerg.mean()):5.2f} max={float(imerg.max()):6.1f}  "
               f"era5 mean={float(era5.mean()):5.2f} max={float(era5.max()):6.1f}")
        return True, msg
    except Exception as exc:  # noqa: BLE001 — keep loop alive on per-day failure
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 78)
    print(f"122-day batch download  |  {config.SEASON_START.date()} → {config.SEASON_END.date()}")
    print(f"Project root: {config.PROJECT_ROOT}")
    print("=" * 78)

    preflight()
    ensure_store()

    todo = missing_dates()
    total = len(todo)
    if total == 0:
        print("[main] All 122 dates already populated. Nothing to do.")
        store.consolidate()
        return 0

    print(f"[main] {total}/122 days need population. Starting...")
    print("[main] Estimated runtime ~3–6 h depending on CDS queue + bandwidth.")
    print()

    failures: list[tuple[pd.Timestamp, str]] = []
    t_start = time.time()

    for i, v in enumerate(todo, start=1):
        t0 = time.time()
        ok, msg = process_one_day(v)
        dt = time.time() - t0
        prefix = f"[{i:3d}/{total}]  {v.date()}  ({dt:5.1f}s)"
        if ok:
            print(f"{prefix}  OK   {msg}")
        else:
            print(f"{prefix}  FAIL {msg}")
            failures.append((v, msg))
        # Flush stdout for tail-able background logs.
        sys.stdout.flush()

    total_min = (time.time() - t_start) / 60
    print()
    print("=" * 78)
    print(f"Summary  |  total time: {total_min:.1f} min")
    print(f"  success:  {total - len(failures)}/{total}")
    print(f"  failed:   {len(failures)}")
    for v, msg in failures:
        print(f"    {v.date()}: {msg}")
    if not failures:
        print("[main] Consolidating Zarr metadata...")
        store.consolidate()
        print("[main] Done.")
    else:
        print("[main] Not consolidating because of failures — re-run to retry failed days.")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
