"""scripts/04b_ingest_aifs.py

Read trimmed AIFS NetCDFs from ``config.FORECAST_DIR``, accumulate each
to an IMD-day total, conservatively regrid to the common 0.25° India
grid, and write to the master Zarr store's ``aifs`` variable.

CPU-only. Idempotent (skips dates whose ``aifs`` is already non-NaN in
the store). Resumable across crashes. Safe to re-run as new forecasts
land on disk.

Run:
    uv run python scripts/04b_ingest_aifs.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from monsoon_bias import config  # noqa: E402
from monsoon_bias.processing import accumulate, regrid, store  # noqa: E402

# Filename: aifs_YYYYMMDDTHHMM_nstepsNN.nc
_INIT_RE = re.compile(r"^aifs_(\d{8}T\d{4})_nsteps(\d+)\.nc$")


def _parse_init(p: Path) -> tuple[pd.Timestamp, int] | None:
    m = _INIT_RE.match(p.name)
    if not m:
        return None
    return pd.Timestamp(m.group(1)), int(m.group(2))


def _verifying_for_init(init: pd.Timestamp) -> pd.Timestamp:
    """init = verifying - lead_days, at 03 UTC → verifying = init.date + lead_days."""
    return init.normalize() + pd.Timedelta(days=config.FORECAST_LEAD_DAYS)


def _already_populated(store_path: Path) -> set[pd.Timestamp]:
    ds = store.open_store(store_path)
    populated = {
        pd.Timestamp(t).normalize()
        for t in ds.time.values
        if not bool(ds.aifs.sel(time=t).isnull().all())
    }
    ds.close()
    return populated


def main() -> int:
    print("=" * 78)
    print("AIFS → Zarr ingest")
    print(f"Forecast dir : {config.FORECAST_DIR}")
    print(f"Zarr store   : {config.ZARR_STORE}")
    print("=" * 78)

    if not config.ZARR_STORE.exists():
        sys.exit(f"[fail] Zarr store missing: {config.ZARR_STORE}. "
                 "Run scripts/03_download_all.py first to create it.")

    # Only trimmed-complete files: exclude .trimmed.nc siblings and very large
    # files (a raw 7 GB write would mean it hasn't been trimmed — skip rather
    # than risk reading a partially-written file).
    candidates = sorted(config.FORECAST_DIR.glob("aifs_*_nsteps*.nc"))
    candidates = [
        p for p in candidates
        if not p.name.endswith(".trimmed.nc")
        and p.stat().st_size < 500_000_000  # 500 MB sanity cap
    ]
    if not candidates:
        print("[main] No AIFS NetCDFs found.")
        return 0
    print(f"[main] {len(candidates)} candidate AIFS NetCDFs on disk.")

    populated = _already_populated(config.ZARR_STORE)
    print(f"[main] {len(populated)} dates already populated in `aifs`.")

    n_written = 0
    n_skipped = 0
    n_outside_season = 0
    failures: list[tuple[str, str]] = []
    t_start = time.time()

    for i, fpath in enumerate(candidates, start=1):
        parsed = _parse_init(fpath)
        if parsed is None:
            print(f"[{i:3d}/{len(candidates)}] skip {fpath.name}: unrecognized filename")
            continue
        init, _nsteps = parsed
        verifying = _verifying_for_init(init)

        if verifying < config.SEASON_START or verifying > config.SEASON_END:
            n_outside_season += 1
            continue
        if verifying in populated:
            n_skipped += 1
            continue

        try:
            with xr.open_dataset(fpath) as ds:
                aifs_native = accumulate.accumulate_aifs_to_imd_day(
                    ds, verifying, init_time=init
                )
            aifs = regrid.regrid_precip(aifs_native)
            store.write_day(verifying, aifs=aifs)
            n_written += 1
            print(f"[{i:3d}/{len(candidates)}] {verifying.date()} ← {fpath.name}  "
                  f"mean={float(aifs.mean()):5.2f} max={float(aifs.max()):6.1f} mm/day  OK")
        except Exception as exc:  # noqa: BLE001 — log + continue
            failures.append((fpath.name, f"{type(exc).__name__}: {exc}"))
            print(f"[{i:3d}/{len(candidates)}] FAIL {fpath.name}: "
                  f"{type(exc).__name__}: {exc}")
        sys.stdout.flush()

    dt = time.time() - t_start
    print()
    print("=" * 78)
    print(f"Summary  |  written: {n_written}  skipped (already done): {n_skipped}  "
          f"outside season: {n_outside_season}  failed: {len(failures)}  |  {dt:.1f}s")
    for name, msg in failures:
        print(f"  FAIL {name}: {msg}")

    if n_written and not failures:
        print("[main] Consolidating Zarr metadata...")
        store.consolidate()
        print("[main] Done.")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
