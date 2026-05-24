"""scripts/06_make_plots.py

CPU. Reads the diagnostic NetCDFs produced by script 05 and writes the
six publication-quality figures into outputs/figures/ at 300 DPI:

  1. mean_bias_map.png            (cmocean.balance, centered at 0)
  2. rmse_map.png                 (cmocean.amp / .rain)
  3. bias_by_bsiso_panels.png     (2x4 small multiples)
  4. bias_vs_elevation.png        (scatter + regression)
  5. bias_by_region_bar.png       (bar)
  6. bias_vs_rainfall_magnitude.png

  STUB: implement after script 05.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from monsoon_bias import config  # noqa: E402


def main() -> int:
    raise NotImplementedError("Implement after script 05.")


if __name__ == "__main__":
    sys.exit(main())
