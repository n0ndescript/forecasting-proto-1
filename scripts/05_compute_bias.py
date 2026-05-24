"""scripts/05_compute_bias.py

CPU. Reads the master Zarr store and writes bias diagnostics:

  - outputs/bias/mean_bias_map.nc
  - outputs/bias/rmse_map.nc
  - outputs/bias/bias_by_bsiso_phase.nc
  - outputs/bias/bias_by_elevation.nc
  - outputs/bias/bias_by_region.nc
  - outputs/bias/bias_by_rainfall_magnitude.nc

All inputs and outputs use mm/day. Land-only mask applied for the
stratifications (see analysis/bias.py).

  STUB: implement once the master Zarr is being populated by 03/04.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from monsoon_bias import config  # noqa: E402


def main() -> int:
    raise NotImplementedError("Implement after the Zarr store is populated.")


if __name__ == "__main__":
    sys.exit(main())
