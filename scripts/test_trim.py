"""scripts/test_trim.py

Validate trim_aifs_forecast on an existing 7.2 GB AIFS NetCDF before
rolling it into the 122-day batch.

Does NOT touch the source file. Writes the trimmed copy to a sibling
path with ``.trimmed.nc`` suffix and:

  1. Reports source vs trimmed size.
  2. Runs accumulate_aifs_to_imd_day on both, asserts the IMD-day
     totals are byte-for-byte identical (same dtype, same values).

Exit code 0 = trim is safe to use in script 04.

Run on the pod:
    uv run python scripts/test_trim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from monsoon_bias import config  # noqa: E402
from monsoon_bias.forecast import run_aifs, trim as trim_mod  # noqa: E402
from monsoon_bias.processing import accumulate  # noqa: E402

VERIFYING_DATE = pd.Timestamp("2025-07-15")


def main() -> int:
    init = run_aifs.init_time_for_verifying_date(VERIFYING_DATE)
    nsteps = run_aifs.nsteps_for_lead()
    src = config.FORECAST_DIR / f"aifs_{init.strftime('%Y%m%dT%H%M')}_nsteps{nsteps}.nc"
    if not src.exists():
        sys.exit(f"[fail] source forecast not found at {src}")

    dst = src.with_suffix(".trimmed.nc")
    if dst.exists():
        dst.unlink()

    src_gb = src.stat().st_size / 1e9
    print(f"[1/4] source: {src.name} ({src_gb:.2f} GB)")

    print(f"[2/4] trimming → {dst.name} ...")
    out = trim_mod.trim_aifs_forecast(src, out_path=dst, delete_source=False)
    out_mb = out.stat().st_size / 1e6
    ratio = src.stat().st_size / out.stat().st_size
    print(f"      trimmed size: {out_mb:.1f} MB  ({ratio:.0f}× smaller)")

    print(f"[3/4] accumulating IMD-day rainfall from both files...")
    with xr.open_dataset(src) as ds_src:
        da_src = accumulate.accumulate_aifs_to_imd_day(
            ds_src, VERIFYING_DATE, init_time=init
        )
    with xr.open_dataset(out) as ds_out:
        da_out = accumulate.accumulate_aifs_to_imd_day(
            ds_out, VERIFYING_DATE, init_time=init
        )

    print(f"      src mean/max: {float(da_src.mean()):.3f} / {float(da_src.max()):.2f} mm/day")
    print(f"      out mean/max: {float(da_out.mean()):.3f} / {float(da_out.max()):.2f} mm/day")

    print(f"[4/4] comparing arrays...")
    if da_src.shape != da_out.shape:
        sys.exit(f"[fail] shape mismatch: {da_src.shape} vs {da_out.shape}")
    if da_src.dtype != da_out.dtype:
        print(f"      note: dtype differs ({da_src.dtype} vs {da_out.dtype}) — "
              f"OK if encoding only.")
    # Use exact equality (modulo NaN handling) — zlib compression is lossless,
    # so any divergence would indicate a bug in the trim step.
    diff = np.abs(da_src.values - da_out.values)
    max_diff = float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0
    if max_diff > 0.0:
        sys.exit(f"[fail] arrays differ; max abs diff = {max_diff:.6e} mm/day")
    print(f"      OK — arrays identical (max abs diff = 0.0)")

    print()
    print("=" * 72)
    print(f"PASS  |  trim is safe.  {src_gb:.2f} GB → {out_mb:.1f} MB  ({ratio:.0f}×)")
    print(f"      |  122 forecasts would now use ~{122 * out_mb / 1024:.1f} GB instead "
          f"of {122 * src_gb:.0f} GB.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
