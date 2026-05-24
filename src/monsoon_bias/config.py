"""Project-wide paths, constants, and domain definitions.

All scripts and modules should import paths and constants from here rather
than hard-coding. Edit this file (not the scripts) when changing the
target season, grid, or domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

ERA5_DIR = DATA_DIR / "era5"
IMERG_DIR = DATA_DIR / "imerg"
FORECAST_DIR = DATA_DIR / "forecasts"
BSISO_DIR = DATA_DIR / "bsiso"
ELEVATION_DIR = DATA_DIR / "elevation"
ZARR_STORE = DATA_DIR / "monsoon_bias.zarr"

# Make sure dirs exist when imported.
for _d in (DATA_DIR, OUTPUTS_DIR, FIGURES_DIR, ERA5_DIR, IMERG_DIR,
          FORECAST_DIR, BSISO_DIR, ELEVATION_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Spatial domain
# ---------------------------------------------------------------------------

# India bounding box. Order: (lat_min, lat_max, lon_min, lon_max).
# Wide enough to cover the Himalayan foothills and the southern peninsula,
# plus the adjacent Arabian Sea and Bay of Bengal for context.
INDIA_BBOX = (6.0, 38.0, 68.0, 98.0)

# Common analysis grid (degrees). 0.25° matches both ERA5 native and the
# IMERG-regridded target chosen for this project.
GRID_RESOLUTION = 0.25

def common_grid() -> dict:
    """Return the 1D lat/lon arrays defining the common 0.25° grid over India."""
    lat_min, lat_max, lon_min, lon_max = INDIA_BBOX
    # Cell centers on the standard 0.25° grid. Inclusive of endpoints.
    lats = np.arange(lat_min, lat_max + GRID_RESOLUTION / 2, GRID_RESOLUTION)
    lons = np.arange(lon_min, lon_max + GRID_RESOLUTION / 2, GRID_RESOLUTION)
    return {"lat": lats, "lon": lons}

# ---------------------------------------------------------------------------
# Time period
# ---------------------------------------------------------------------------

# 2025 Indian monsoon season. Today is 2026-05-19, so IMERG Final V07
# (~3.5-month latency) and ERA5 final (~3-month latency) both fully cover
# this window.
SEASON_START = pd.Timestamp("2025-06-01")
SEASON_END = pd.Timestamp("2025-09-30")

# Forecast lead time. We initialize from ERA5 3 days before the verifying
# date and verify on a single 24-hour accumulation window aligned to IMD.
FORECAST_LEAD_DAYS = 3

# IMD "daily rainfall" runs 08:30 IST → 08:30 IST (next day).
# IST = UTC + 5:30, so the window is 03:00 UTC → 03:00 UTC (next day).
IMD_DAY_START_HOUR_UTC = 3

def imd_day_window_utc(verifying_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """For a given IMD calendar date, return the UTC half-open window
    [start, end) covering the IMD meteorological day.

    Example: verifying_date = 2025-07-15 →
        (2025-07-15 03:00 UTC, 2025-07-16 03:00 UTC)
    """
    start = pd.Timestamp(verifying_date).normalize() + pd.Timedelta(hours=IMD_DAY_START_HOUR_UTC)
    end = start + pd.Timedelta(days=1)
    return start, end

def season_dates() -> pd.DatetimeIndex:
    """All IMD verifying dates in the target monsoon season."""
    return pd.date_range(SEASON_START, SEASON_END, freq="D")

# ---------------------------------------------------------------------------
# Stratification bins
# ---------------------------------------------------------------------------

# BSISO active phases per Kikuchi index (1..8). Phase 0 = inactive (low
# amplitude) and is handled separately.
BSISO_PHASES = (1, 2, 3, 4, 5, 6, 7, 8)
BSISO_AMPLITUDE_THRESHOLD = 1.0  # standard convention for "active"

# Elevation bins (meters). Captures plain → foothills → low mountain →
# high Himalaya.
ELEVATION_BINS_M = (0, 500, 1500, 3000, 9000)
ELEVATION_LABELS = ("plain", "foothill", "low_mountain", "high_mountain")

# Observed rainfall magnitude bins (mm/day, IMD convention).
RAINFALL_BINS_MM = (0, 1, 10, 35, 75, 1000)
RAINFALL_LABELS = ("trace", "light", "moderate", "heavy", "very_heavy")


@dataclass(frozen=True)
class Region:
    """Approximate rectangular region of meteorological interest.

    These are rough bounding boxes meant for first-cut stratification;
    proper polygons would use a shapefile (e.g., from IMD subdivisions or
    Natural Earth) — easy to swap in later.
    """
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


REGIONS: tuple[Region, ...] = (
    # Western Ghats windward face — narrow strip on the west of the range.
    Region("western_ghats_windward", 8.0, 21.0, 73.0, 75.0),
    # Western Ghats leeward (rain-shadow side).
    Region("western_ghats_leeward", 8.0, 21.0, 75.0, 77.5),
    # Indo-Gangetic plain.
    Region("gangetic_plain", 24.0, 30.0, 76.0, 88.0),
    # Northeast India (Brahmaputra valley + Meghalaya).
    Region("northeast", 22.0, 29.0, 88.0, 97.0),
    # Himalayan foothills.
    Region("himalayan_foothills", 28.0, 32.0, 75.0, 95.0),
    # Peninsular interior (Deccan).
    Region("peninsular_interior", 12.0, 22.0, 76.0, 83.0),
)
