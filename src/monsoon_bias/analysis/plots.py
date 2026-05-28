"""Publication-quality plots.

All figures are saved as 300-DPI PNGs into :data:`config.FIGURES_DIR`.
Maps use cartopy (PlateCarree) with coastlines and Indian state
boundaries. Colormaps come from cmocean (``balance`` diverging for bias
maps; ``amp`` sequential for RMSE).

Each plot takes the result of a function in :mod:`monsoon_bias.analysis.bias`
plus an output path. They return the saved Path.
"""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .. import config

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PROJ = ccrs.PlateCarree()
_DEFAULT_DPI = 300


def _india_extent() -> tuple[float, float, float, float]:
    """(lon_min, lon_max, lat_min, lat_max) — matplotlib extent convention."""
    lat_min, lat_max, lon_min, lon_max = config.INDIA_BBOX
    return lon_min, lon_max, lat_min, lat_max


def _map_axes(ax) -> None:
    """Decorate a cartopy axes with coastlines, country borders, state lines, gridlines."""
    ax.set_extent(_india_extent(), crs=_PROJ)
    ax.coastlines(linewidth=0.6, color="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black")
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "cultural", "admin_1_states_provinces_lines", "10m"
        ),
        linewidth=0.3,
        edgecolor="gray",
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, color="gray")
    gl.top_labels = False
    gl.right_labels = False


def _symmetric_lim(da: xr.DataArray, q: float = 0.98) -> float:
    """Symmetric vmin/vmax for diverging colormaps. Clips at the q-th quantile
    of |values| so a few outlier cells don't compress the rest."""
    vals = np.abs(da.values[np.isfinite(da.values)])
    if vals.size == 0:
        return 1.0
    return float(np.quantile(vals, q))


def _ensure_outdir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _savefig(fig: plt.Figure, output_path: Path) -> Path:
    _ensure_outdir(output_path)
    fig.savefig(output_path, dpi=_DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

def plot_mean_bias_map(
    bias_map: xr.DataArray,
    output_path: Path,
    *,
    title: str | None = None,
    vmax: float | None = None,
) -> Path:
    """Diverging-colormap map of mean bias (cmocean.cm.balance)."""
    if vmax is None:
        vmax = _symmetric_lim(bias_map)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1, projection=_PROJ)
    _map_axes(ax)
    mesh = ax.pcolormesh(
        bias_map.lon, bias_map.lat, bias_map.values,
        cmap=cmocean.cm.balance, vmin=-vmax, vmax=vmax,
        transform=_PROJ, shading="nearest",
    )
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.7, label="mm/day")
    n_days = bias_map.attrs.get("n_days", "?")
    ax.set_title(title or f"{bias_map.attrs.get('long_name', 'mean bias')}  (n_days={n_days})")
    return _savefig(fig, output_path)


def plot_rmse_map(
    rmse_map: xr.DataArray,
    output_path: Path,
    *,
    title: str | None = None,
    vmax: float | None = None,
) -> Path:
    """Sequential-colormap map of RMSE (cmocean.cm.amp)."""
    if vmax is None:
        vmax = float(np.quantile(rmse_map.values[np.isfinite(rmse_map.values)], 0.98))
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1, projection=_PROJ)
    _map_axes(ax)
    mesh = ax.pcolormesh(
        rmse_map.lon, rmse_map.lat, rmse_map.values,
        cmap=cmocean.cm.amp, vmin=0, vmax=vmax,
        transform=_PROJ, shading="nearest",
    )
    fig.colorbar(mesh, ax=ax, shrink=0.7, label="mm/day")
    n_days = rmse_map.attrs.get("n_days", "?")
    ax.set_title(title or f"{rmse_map.attrs.get('long_name', 'RMSE')}  (n_days={n_days})")
    return _savefig(fig, output_path)


def plot_bias_three_panel(
    aifs_bias: xr.DataArray,
    era5_bias: xr.DataArray,
    residual: xr.DataArray,
    output_path: Path,
    *,
    title: str | None = None,
    vmax: float | None = None,
) -> Path:
    """Three-panel comparison: AIFS−IMERG | ERA5−IMERG | AIFS−ERA5.

    Shared symmetric color scale (so panels are visually comparable).
    """
    if vmax is None:
        vmax = max(_symmetric_lim(aifs_bias), _symmetric_lim(era5_bias), _symmetric_lim(residual))
    fig = plt.figure(figsize=(18, 6))
    panels = (
        ("AIFS − IMERG", aifs_bias),
        ("ERA5 − IMERG", era5_bias),
        ("AIFS − ERA5 (residual)", residual),
    )
    for i, (label, da) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i, projection=_PROJ)
        _map_axes(ax)
        mesh = ax.pcolormesh(
            da.lon, da.lat, da.values,
            cmap=cmocean.cm.balance, vmin=-vmax, vmax=vmax,
            transform=_PROJ, shading="nearest",
        )
        ax.set_title(label, fontsize=11)
        if i == 3:
            fig.colorbar(mesh, ax=ax, shrink=0.7, label="mm/day")
    if title:
        fig.suptitle(title, fontsize=13)
    return _savefig(fig, output_path)


# ---------------------------------------------------------------------------
# Stratification charts
# ---------------------------------------------------------------------------

def plot_bias_by_region(
    region_stats: xr.Dataset,
    output_path: Path,
    *,
    title: str | None = None,
) -> Path:
    """Horizontal bar chart: mean bias (left) + RMSE (right) per region."""
    regions = list(region_stats.region.values)
    bias = region_stats.bias.values
    rmse = region_stats.rmse.values
    y = np.arange(len(regions))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    axes[0].barh(y, bias, color=["#c0392b" if b > 0 else "#2980b9" for b in bias])
    axes[0].axvline(0, color="black", linewidth=0.6)
    axes[0].set_yticks(y, regions)
    axes[0].set_xlabel("mean bias (mm/day)")
    axes[0].set_title("bias")
    axes[0].invert_yaxis()

    axes[1].barh(y, rmse, color="#7f8c8d")
    axes[1].set_xlabel("RMSE (mm/day)")
    axes[1].set_title("RMSE")

    fcst = region_stats.attrs.get("forecast", "?")
    obs = region_stats.attrs.get("observed", "?")
    fig.suptitle(title or f"bias by region ({fcst} vs {obs})")
    fig.tight_layout()
    return _savefig(fig, output_path)


def plot_bias_by_rainfall_magnitude(
    rain_stats: xr.Dataset,
    output_path: Path,
    *,
    title: str | None = None,
) -> Path:
    """Bar chart: mean bias per observed-rainfall-magnitude bin."""
    labels = list(rain_stats.rain_bin.values)
    bias = rain_stats.bias.values
    counts = rain_stats["count"].values
    los = rain_stats.rain_bin_lo_mm.values
    his = rain_stats.rain_bin_hi_mm.values
    annotated = [f"{lab}\n({lo:g}–{hi:g} mm)" for lab, lo, hi in zip(labels, los, his)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        annotated, bias,
        color=["#c0392b" if b > 0 else "#2980b9" for b in bias],
        edgecolor="black", linewidth=0.5,
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("mean bias (mm/day)")
    ax.set_xlabel("observed rainfall bin")
    # Annotate with sample count
    for bar, n in zip(bars, counts):
        ymin, ymax = ax.get_ylim()
        y_text = bar.get_height() + (ymax - ymin) * 0.02 if bar.get_height() >= 0 else bar.get_height() - (ymax - ymin) * 0.04
        ax.text(bar.get_x() + bar.get_width() / 2, y_text, f"n={n:,}",
                ha="center", va="bottom" if bar.get_height() >= 0 else "top", fontsize=8)
    fcst = rain_stats.attrs.get("forecast", "?")
    obs = rain_stats.attrs.get("observed", "?")
    ax.set_title(title or f"bias by observed-rainfall magnitude ({fcst} vs {obs})")
    fig.tight_layout()
    return _savefig(fig, output_path)


def plot_bias_by_elevation(
    elev_stats: xr.Dataset,
    output_path: Path,
    *,
    title: str | None = None,
) -> Path:
    """Bar chart: mean bias per elevation bin."""
    labels = list(elev_stats.elev_bin.values)
    bias = elev_stats.bias.values
    counts = elev_stats["count"].values
    los = elev_stats.elev_bin_lo_m.values
    his = elev_stats.elev_bin_hi_m.values
    annotated = [f"{lab}\n({lo:g}–{hi:g} m)" for lab, lo, hi in zip(labels, los, his)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        annotated, bias,
        color=["#c0392b" if b > 0 else "#2980b9" for b in bias],
        edgecolor="black", linewidth=0.5,
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("mean bias (mm/day)")
    ax.set_xlabel("elevation bin")
    for bar, n in zip(bars, counts):
        ymin, ymax = ax.get_ylim()
        y_text = bar.get_height() + (ymax - ymin) * 0.02 if bar.get_height() >= 0 else bar.get_height() - (ymax - ymin) * 0.04
        ax.text(bar.get_x() + bar.get_width() / 2, y_text, f"n={n:,}",
                ha="center", va="bottom" if bar.get_height() >= 0 else "top", fontsize=8)
    fcst = elev_stats.attrs.get("forecast", "?")
    obs = elev_stats.attrs.get("observed", "?")
    ax.set_title(title or f"bias by elevation ({fcst} vs {obs})")
    fig.tight_layout()
    return _savefig(fig, output_path)


def plot_bias_vs_elevation_scatter(
    bias_map: xr.DataArray,
    elevation: xr.DataArray,
    output_path: Path,
    *,
    title: str | None = None,
) -> Path:
    """Hexbin of pointwise mean-bias vs elevation, with a linear regression line."""
    if bias_map.dims != ("lat", "lon"):
        raise ValueError(f"bias_map must be (lat, lon), got {bias_map.dims}")
    if elevation.dims != ("lat", "lon"):
        raise ValueError(f"elevation must be (lat, lon), got {elevation.dims}")
    b = bias_map.values.ravel()
    e = elevation.values.ravel()
    finite = np.isfinite(b) & np.isfinite(e)
    b, e = b[finite], e[finite]

    fig, ax = plt.subplots(figsize=(8, 5))
    hb = ax.hexbin(e, b, gridsize=40, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax, label="cells per bin")
    ax.axhline(0, color="white", linewidth=0.8, alpha=0.7)

    # Simple linear fit
    if len(e) >= 2:
        slope, intercept = np.polyfit(e, b, 1)
        e_line = np.array([e.min(), e.max()])
        ax.plot(e_line, slope * e_line + intercept, color="red", linewidth=1.2,
                label=f"fit: {slope * 1000:+.2f} mm/day per km")
        ax.legend(loc="upper right")

    ax.set_xlabel("elevation (m)")
    ax.set_ylabel("mean bias (mm/day)")
    ax.set_title(title or "Mean bias vs elevation (pointwise)")
    fig.tight_layout()
    return _savefig(fig, output_path)


# ---------------------------------------------------------------------------
# Deferred
# ---------------------------------------------------------------------------

def plot_bias_by_bsiso(bias_by_phase, output_path: Path):
    """Small-multiples: 2x4 panel of bias maps, one per BSISO phase, with a
    shared color scale. Deferred — see STATUS.md.
    """
    raise NotImplementedError(
        "BSISO stratification is deferred — no live index source. "
        "See STATUS.md → Known limitations."
    )
