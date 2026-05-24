"""scripts/02_download_one_date.py

Full pipeline for ONE verifying date, end-to-end. Run this until it
works before attempting the 122-day batch (script 03).

Steps (CPU = laptop, GPU = needs CUDA):

  1. [CPU] Download IMERG half-hourly granules for the IMD day.
  2. [CPU] Download ERA5 hourly tp for the IMD day (baseline panel).
  3. [GPU] Run AIFS for 96 h from a 03 UTC init three days back.
            Skipped with a clear message if no CUDA.
  4. [CPU] Accumulate IMERG → mm/day (sum of 48 half-hours).
  5. [CPU] Accumulate ERA5  → mm/day (sum of 24 hourly tp).
  6. [CPU] (if step 3 ran) Accumulate AIFS → mm/day (sum of 4 tp06 steps).
  7. [CPU] Conservatively regrid all to the common 0.25° grid.
  8. [CPU] Plot 3 panels (IMERG, ERA5, AIFS if present) + the bias maps.

Runs with no CLI args:   python scripts/02_download_one_date.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from monsoon_bias import config  # noqa: E402
from monsoon_bias.data import era5 as era5_dl, imerg as imerg_dl  # noqa: E402
from monsoon_bias.processing import accumulate, regrid  # noqa: E402

VERIFYING_DATE = pd.Timestamp("2025-07-15")
FIG_PATH = config.FIGURES_DIR / f"02_one_date_{VERIFYING_DATE.date()}.png"


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _step(name: str) -> None:
    print(f"\n[{name}]")


def fetch_imerg() -> xr.DataArray:
    _step("IMERG")
    print(f"  Downloading {imerg_dl.EXPECTED_GRANULES_PER_IMD_DAY} half-hourly granules for IMD day {VERIFYING_DATE.date()}...")
    paths = imerg_dl.download_imerg_for_imd_day(VERIFYING_DATE)
    print(f"  Got {len(paths)} files; accumulating to mm/day...")
    da = accumulate.accumulate_imerg_half_hourly_to_imd_day(paths, VERIFYING_DATE)
    print(f"  OK — IMERG mean over India: {float(da.mean()):.2f} mm/day, "
          f"max: {float(da.max()):.1f} mm/day")
    return da


def fetch_era5() -> xr.DataArray:
    _step("ERA5 baseline")
    print(f"  Downloading hourly ERA5 tp for the IMD day...")
    path = era5_dl.download_era5_precip_imd_day(VERIFYING_DATE)
    print(f"  Got {path.name} ({path.stat().st_size / 1024:.0f} KB); accumulating...")
    ds = xr.open_dataset(path)
    # Newer CDS files name the time coord 'valid_time'; older use 'time'.
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    # Standardize lat/lon names.
    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    da = accumulate.accumulate_era5_hourly_to_imd_day(ds, VERIFYING_DATE, var="tp")
    print(f"  OK — ERA5 mean over India: {float(da.mean()):.2f} mm/day, "
          f"max: {float(da.max()):.1f} mm/day")
    return da


def fetch_aifs() -> xr.DataArray | None:
    """Run AIFS if a GPU is available; otherwise skip with a clear note."""
    _step("AIFS forecast")
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  Skipped — torch not installed. Install with `uv sync --extra gpu` on the GPU host.")
        return None

    import torch
    if not torch.cuda.is_available():
        print("  Skipped — no CUDA device. Re-run on the H100 host or set up "
              "torch with CUDA locally.")
        return None

    try:
        from monsoon_bias.forecast import run_aifs
    except ImportError as exc:
        print(f"  Skipped — earth2studio not installed: {exc}")
        return None

    print(f"  Running AIFS for verifying date {VERIFYING_DATE.date()} "
          f"(init {run_aifs.init_time_for_verifying_date(VERIFYING_DATE)})...")
    fcst_path = run_aifs.run_aifs_forecast(VERIFYING_DATE)
    print(f"  Got {fcst_path.name}; extracting IMD-day rainfall...")
    ds = xr.open_dataset(fcst_path)
    init = run_aifs.init_time_for_verifying_date(VERIFYING_DATE)
    da = accumulate.accumulate_aifs_to_imd_day(ds, VERIFYING_DATE, init_time=init)
    print(f"  OK — AIFS mean over India: {float(da.mean()):.2f} mm/day, "
          f"max: {float(da.max()):.1f} mm/day")
    return da


def regrid_all(*fields: tuple[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    _step("Regrid")
    out = {}
    for name, da in fields:
        if da is None:
            continue
        print(f"  Regridding {name} (source shape {da.shape}) → common 0.25° grid...")
        out[name] = regrid.regrid_precip(da)
    # Sanity: all outputs share the same target grid.
    names = list(out)
    for n in names[1:]:
        regrid.verify_coastline_alignment(out[names[0]], out[n])
    print(f"  OK — all fields aligned on {out[names[0]].shape} grid.")
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _setup_india_axis(ax) -> None:
    import cartopy.feature as cfeature
    lat_min, lat_max, lon_min, lon_max = config.INDIA_BBOX
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle="--")
    try:
        states = cfeature.NaturalEarthFeature(
            "cultural", "admin_1_states_provinces_lines", "10m",
            facecolor="none", edgecolor="gray", linewidth=0.25)
        ax.add_feature(states)
    except Exception:  # noqa: BLE001
        pass
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False


def plot(fields: dict[str, xr.DataArray]) -> Path:
    import cartopy.crs as ccrs
    import cmocean

    proj = ccrs.PlateCarree()
    names = [n for n in ("imerg", "era5", "aifs") if n in fields]
    bias_pairs = [(other, "imerg") for other in ("era5", "aifs") if other in fields]
    n_top = len(names)
    n_bot = len(bias_pairs)

    fig, axes = plt.subplots(
        2, max(n_top, n_bot), figsize=(5.5 * max(n_top, n_bot), 11),
        subplot_kw={"projection": proj}, constrained_layout=True,
    )
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    # Row 1: rainfall fields on a shared scale.
    vmax = max(float(np.nanpercentile(fields[n].values, 99)) for n in names)
    vmax = max(50.0, vmax)
    rain_cmap = cmocean.cm.rain
    for i, n in enumerate(names):
        ax = axes[0, i]
        _setup_india_axis(ax)
        da = fields[n]
        data = np.where(da.values < 0.1, np.nan, da.values)
        mesh = ax.pcolormesh(da["lon"], da["lat"], data, transform=proj,
                             cmap=rain_cmap, shading="auto", vmin=0, vmax=vmax)
        cb = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.85)
        cb.set_label("mm/day")
        ax.set_title({"imerg": "IMERG Final V07 (obs)",
                      "era5":  "ERA5 (reanalysis baseline)",
                      "aifs":  "AIFS forecast (3-day lead)"}[n])
    # Hide unused top-row axes.
    for i in range(n_top, axes.shape[1]):
        axes[0, i].axis("off")

    # Row 2: bias maps (forecast/baseline − obs) on a diverging scale.
    bias_cmap = cmocean.cm.balance
    bias_max = 0.0
    bias_arrays = {}
    for a, b in bias_pairs:
        bias_arrays[(a, b)] = fields[a] - fields[b]
        bias_max = max(bias_max, float(np.nanpercentile(np.abs(bias_arrays[(a, b)].values), 99)))
    bias_max = max(10.0, bias_max)
    for i, (a, b) in enumerate(bias_pairs):
        ax = axes[1, i]
        _setup_india_axis(ax)
        d = bias_arrays[(a, b)]
        mesh = ax.pcolormesh(d["lon"], d["lat"], d.values, transform=proj,
                             cmap=bias_cmap, shading="auto",
                             vmin=-bias_max, vmax=bias_max)
        cb = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.85)
        cb.set_label(f"{a.upper()} − {b.upper()}  (mm/day)")
        ax.set_title(f"Bias: {a.upper()} − {b.upper()}")
    for i in range(n_bot, axes.shape[1]):
        axes[1, i].axis("off")

    fig.suptitle(
        f"One-date pipeline — IMD day {VERIFYING_DATE.date()}  "
        f"(03 UTC → 03 UTC next day)", fontsize=13,
    )
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200)
    plt.close(fig)
    return FIG_PATH


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print(f"One-date pipeline  |  verifying IMD day: {VERIFYING_DATE.date()}")
    print(f"Project root: {config.PROJECT_ROOT}")
    print("=" * 72)

    imerg = fetch_imerg()
    era5 = fetch_era5()
    aifs = fetch_aifs()

    fields_native = [("imerg", imerg), ("era5", era5)]
    if aifs is not None:
        fields_native.append(("aifs", aifs))
    fields = regrid_all(*fields_native)

    _step("Plot")
    out = plot(fields)
    print(f"  Saved {out}")

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    for name in fields:
        da = fields[name]
        print(f"  {name:6s}: mean {float(da.mean()):6.2f}  max {float(da.max()):6.1f} mm/day")
    if "aifs" not in fields:
        print("  (AIFS skipped — re-run on the GPU host to include the forecast panel.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
