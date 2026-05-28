"""scripts/06_make_plots.py

CPU. Reads bias diagnostics produced by ``scripts/05_compute_bias.py``
and writes publication-quality figures to ``config.FIGURES_DIR``:

    01_mean_bias_aifs.png
    02_rmse_aifs.png
    03_three_panel_aifs_era5_residual.png
    04_bias_by_region_aifs.png
    05_bias_by_elevation_aifs.png
    06_bias_vs_elevation_scatter_aifs.png
    07_bias_by_rainfall_magnitude_aifs.png

Run:
    uv run python scripts/06_make_plots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import xarray as xr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from monsoon_bias import config  # noqa: E402
from monsoon_bias.analysis import plots  # noqa: E402

BIAS_DIR = config.OUTPUTS_DIR / "bias"
FIG_DIR = config.FIGURES_DIR


def _load_da(name: str) -> xr.DataArray:
    return xr.open_dataarray(BIAS_DIR / name)


def _load_ds(name: str) -> xr.Dataset:
    return xr.open_dataset(BIAS_DIR / name)


def main() -> int:
    print("=" * 72)
    print("Make plots")
    print(f"Bias dir : {BIAS_DIR}")
    print(f"Fig dir  : {FIG_DIR}")
    print("=" * 72)

    if not BIAS_DIR.exists():
        sys.exit(f"[fail] {BIAS_DIR} missing. Run scripts/05_compute_bias.py first.")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    aifs_bias = _load_da("mean_bias_map_aifs.nc")
    aifs_rmse = _load_da("rmse_map_aifs.nc")
    era5_bias = _load_da("mean_bias_map_era5.nc")
    residual = _load_da("mean_bias_map_residual.nc")
    region_stats = _load_ds("bias_by_region.nc")
    elev_stats = _load_ds("bias_by_elevation.nc")
    rain_stats = _load_ds("bias_by_rainfall_magnitude.nc")
    elev_field = _load_da("_elevation_on_common_grid.nc")

    n_days = aifs_bias.attrs.get("n_days", "?")
    print(f"[main] n_days (AIFS): {n_days}")

    print("[1/7] mean bias map (AIFS)")
    plots.plot_mean_bias_map(
        aifs_bias,
        FIG_DIR / "01_mean_bias_aifs.png",
        title=f"AIFS − IMERG mean bias  (n_days={n_days})",
    )

    print("[2/7] RMSE map (AIFS)")
    plots.plot_rmse_map(
        aifs_rmse,
        FIG_DIR / "02_rmse_aifs.png",
        title=f"AIFS vs IMERG RMSE  (n_days={n_days})",
    )

    print("[3/7] three-panel: AIFS − IMERG | ERA5 − IMERG | AIFS − ERA5")
    plots.plot_bias_three_panel(
        aifs_bias, era5_bias, residual,
        FIG_DIR / "03_three_panel_aifs_era5_residual.png",
        title=f"Bias decomposition  (n_days={n_days})",
    )

    print("[4/7] bias by region (AIFS)")
    plots.plot_bias_by_region(
        region_stats,
        FIG_DIR / "04_bias_by_region_aifs.png",
    )

    print("[5/7] bias by elevation bin (AIFS)")
    plots.plot_bias_by_elevation(
        elev_stats,
        FIG_DIR / "05_bias_by_elevation_aifs.png",
    )

    print("[6/7] bias vs elevation scatter (AIFS)")
    plots.plot_bias_vs_elevation_scatter(
        aifs_bias, elev_field,
        FIG_DIR / "06_bias_vs_elevation_scatter_aifs.png",
        title=f"AIFS mean bias vs elevation  (n_days={n_days})",
    )

    print("[7/7] bias by rainfall magnitude (AIFS)")
    plots.plot_bias_by_rainfall_magnitude(
        rain_stats,
        FIG_DIR / "07_bias_by_rainfall_magnitude_aifs.png",
    )

    print()
    print(f"[main] Wrote 7 figures to {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
