"""scripts/04_run_forecasts.py

GPU-bound: run AIFS forecasts for every verifying date in the season.

For each verifying date d in config.season_dates(), runs a 96-h AIFS
forecast initialized at (d - 3 days, 03 UTC) using ARCO-ERA5 as the IC
source. One NetCDF per init time, written to config.FORECAST_DIR.

Resumable. Skips dates whose output NetCDF already exists. Logs
per-forecast wall-clock and any failures; failures don't abort the loop.

Pre-flight: refuses to run on a CPU-only host.

Run:
    uv run python scripts/04_run_forecasts.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from monsoon_bias import config  # noqa: E402
from monsoon_bias.forecast import run_aifs  # noqa: E402


def main() -> int:
    print("=" * 78)
    print(f"AIFS batch forecasts | {config.SEASON_START.date()} → {config.SEASON_END.date()}")
    print(f"Project root: {config.PROJECT_ROOT}")
    print("=" * 78)

    # Pre-flight: must be on a GPU.
    try:
        import torch
    except ImportError:
        sys.exit("[preflight] torch not installed. Run `uv sync --extra gpu` first.")
    if not torch.cuda.is_available():
        sys.exit("[preflight] No CUDA device. This script needs a GPU host.")
    p = torch.cuda.get_device_properties(0)
    print(f"[preflight] {torch.cuda.get_device_name(0)} "
          f"({p.total_memory / 1024**3:.1f} GB, sm_{p.major}{p.minor})")

    dates = config.season_dates()
    nsteps = run_aifs.nsteps_for_lead()

    todo: list[pd.Timestamp] = []
    for v in dates:
        init = run_aifs.init_time_for_verifying_date(v)
        path = config.FORECAST_DIR / f"aifs_{init.strftime('%Y%m%dT%H%M')}_nsteps{nsteps}.nc"
        if not path.exists():
            todo.append(v)

    total = len(todo)
    if total == 0:
        print("[main] All forecasts already exist on disk. Nothing to do.")
        return 0
    print(f"[main] {total}/{len(dates)} forecasts to run. Starting...")
    print()

    failures: list[tuple[pd.Timestamp, str]] = []
    t_start = time.time()

    for i, v in enumerate(todo, start=1):
        t0 = time.time()
        try:
            path = run_aifs.run_aifs_forecast(v)
            dt = time.time() - t0
            print(f"[{i:3d}/{total}]  verifying {v.date()}  ({dt:5.1f}s)  OK  {path.name}")
        except Exception as exc:  # noqa: BLE001 — keep loop alive on per-day failure
            dt = time.time() - t0
            print(f"[{i:3d}/{total}]  verifying {v.date()}  ({dt:5.1f}s)  FAIL  {type(exc).__name__}: {exc}")
            failures.append((v, str(exc)))
        sys.stdout.flush()

    total_min = (time.time() - t_start) / 60
    print()
    print("=" * 78)
    print(f"Summary | wall-clock {total_min:.1f} min | "
          f"{total - len(failures)}/{total} succeeded")
    for v, msg in failures:
        print(f"  FAIL {v.date()}: {msg}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
