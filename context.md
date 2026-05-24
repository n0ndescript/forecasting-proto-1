# Project: AI Weather Model Bias Analysis over India

## Context

I'm prototyping a research project evaluating whether off-the-shelf global
AI weather models (specifically ECMWF's AIFS and Google's GraphCast) have
systematic, geographically-coherent biases when applied to the Indian
monsoon region. If such biases exist with clear structure, this motivates
building "corrective heads" — small ML models that learn to correct the
global model's outputs against local observations.

This prototype tests the precondition: **does the bias have learnable structure?**
It does NOT build the corrective head itself. That's the follow-on project.

## What I need you to build

A reproducible Python pipeline that:

1. Downloads ERA5 initial conditions from Copernicus CDS for a specified
   date range during the 2024 Indian monsoon (June 1 – Sept 30, 2024)
2. Runs a 3-day forecast using AIFS (or GraphCast) via NVIDIA Earth2Studio
3. Extracts 24-hour accumulated rainfall valid at the verifying date,
   aligned to the IMD meteorological day (08:30 IST to 08:30 IST)
4. Downloads matching observational ground truth from NASA GPM IMERG
5. Regrids forecasts and observations to a common 0.25° grid using
   conservative regridding (xesmf)
6. Saves everything to a single Zarr store with consistent dimensions
7. Computes bias diagnostics: mean bias map, RMSE map, stratifications by
   BSISO phase, elevation, region, and rainfall magnitude
8. Produces publication-quality plots

## Project structure I want
monsoon-bias-prototype/
├── README.md
├── pyproject.toml          # use uv or poetry, not raw pip
├── .gitignore              # exclude data/, outputs/, .env
├── .env.example            # CDS_API_KEY, EARTHDATA_USERNAME, etc.
├── src/
│   └── monsoon_bias/
│       ├── init.py
│       ├── config.py       # paths, constants, BBOXES, time periods
│       ├── data/
│       │   ├── era5.py     # ERA5 downloader via cdsapi
│       │   ├── imerg.py    # IMERG downloader via NASA Earthdata
│       │   ├── bsiso.py    # BSISO index downloader (Kikuchi)
│       │   └── elevation.py # ETOPO1 or SRTM for orographic analysis
│       ├── forecast/
│       │   └── run_aifs.py # Earth2Studio forecast pipeline
│       ├── processing/
│       │   ├── regrid.py    # conservative regridding via xesmf
│       │   ├── accumulate.py # 24h rainfall aligned to IMD day
│       │   └── store.py     # Zarr I/O
│       └── analysis/
│           ├── bias.py      # mean bias, RMSE, stratifications
│           └── plots.py     # cmocean colormaps, cartopy maps
├── scripts/
│   ├── 01_test_credentials.py   # verify CDS + Earthdata access
│   ├── 02_download_one_date.py  # full pipeline for ONE date end-to-end
│   ├── 03_download_all.py       # production: 122 days
│   ├── 04_run_forecasts.py      # GPU-bound: runs on RunPod
│   ├── 05_compute_bias.py       # CPU: analysis
│   └── 06_make_plots.py         # CPU: visualization
└── notebooks/
├── exploration.ipynb         # scratch space
└── final_writeup.ipynb       # generates plots for synthesis doc
## Constraints and conventions

**Environment management:**
- Use `uv` for dependency management (faster than pip, modern)
- Python 3.11
- All scripts must be runnable as `python scripts/XX_name.py` with no
  required CLI args (use config.py for paths)

**Data handling:**
- Use xarray for everything. Never raw numpy arrays for gridded data.
- Use Zarr for the master store. NetCDF for individual file outputs.
- Coordinates should be named `time`, `lat`, `lon` (not latitude/longitude).
- All datasets should have explicit CF-compliant metadata where possible.

**Grid alignment is CRITICAL:**
- Use xesmf with `method="conservative"` for rainfall regridding.
- Bilinear regridding is WRONG for precipitation — it doesn't preserve
  total water. Refuse to do it for rainfall fields.
- After regridding, verify alignment by checking that coastlines match
  between the two fields.

**Accumulation windows are CRITICAL:**
- IMD's "daily rainfall" runs 08:30 IST to 08:30 IST (which is 03:00 UTC
  to 03:00 UTC the next day).
- ERA5 hourly precipitation needs to be summed over this window.
- IMERG half-hourly needs to be summed over the same window.
- Document the conversion explicitly in code comments.

**Avoid these common bugs:**
- Don't use forecast lead time naively — verify the forecast initialized
  3 days before the verifying date actually produces output at the correct
  valid time.
- Don't average over NaN-filled grid points (use `.mean(dim="time", skipna=True)`).
- Don't compute bias before checking that both fields have the same units
  (mm/day, not mm/s or kg/m²).
- Don't trust the model's land-sea mask — apply your own mask if comparing
  only over land.

**Spatial constraints:**
- India bounding box: lat 6°N to 38°N, lon 68°E to 98°E.
- Keep ocean grid points in the dataset for context, but compute land-only
  statistics for the bias analysis (since IMERG over ocean is different
  quality than over land).

## Specific things I want produced

### From scripts/01_test_credentials.py
A tiny end-to-end test that downloads:
- 1 day of ERA5 2m temperature over India (~50 KB)
- 1 day of IMERG rainfall over India (~few MB)
- Plots both on a map of India with cartopy + state boundaries
- Confirms my credentials work and visualizes the data is sensible

### From scripts/02_download_one_date.py
The full pipeline for ONE date (e.g., July 15, 2024) end-to-end:
1. Download ERA5 initial conditions for July 12, 2024 00 UTC
2. Run AIFS for 3 days
3. Extract 24-hour rainfall valid 03:00 UTC July 15 → 03:00 UTC July 16
4. Download IMERG for the same window
5. Regrid both to common grid
6. Compute bias
7. Plot forecast, observation, and bias side-by-side

This script proves the pipeline works before scaling to 122 days.
Critical for debugging — don't move on until this works correctly.

### From scripts/05_compute_bias.py
Computes and saves to disk:
- mean_bias_map (lat, lon)
- rmse_map (lat, lon)
- bias_by_bsiso_phase (phase, lat, lon)   # 8 panels
- bias_by_elevation (binned: <500m, 500-1500m, 1500-3000m, >3000m)
- bias_by_region (Western Ghats windward, leeward, Gangetic plain,
                  Northeast, Himalayan foothills, peninsular interior)
- bias_by_rainfall_magnitude (binned by observed: <1, 1-10, 10-35, 35-75, >75 mm)

### From scripts/06_make_plots.py
Six publication-quality figures using cmocean colormaps:
1. Mean bias map of India (diverging colormap, centered at 0)
2. RMSE map of India (sequential colormap)
3. Small-multiples panel: bias map for each of 8 BSISO phases (shared color scale)
4. Scatter plot: bias vs elevation, with regression line
5. Bar chart: mean bias and RMSE by region
6. Plot: bias as a function of observed rainfall magnitude

All plots:
- Use cartopy for maps with state boundaries and coastlines
- 300 DPI PNG output
- Consistent fonts (whatever cartopy's default is fine)
- Save to `outputs/figures/`

## Important: how to work

1. **Build incrementally.** Don't write the entire pipeline before testing
   any of it. Get `01_test_credentials.py` working first. Then
   `02_download_one_date.py`. Don't move to 122 days until 1 day is rock solid.

2. **No mocks or fake data.** If you can't actually call the API yet
   because you don't have my key, write the code that WOULD work and add
   a clear `TODO: tested when credentials available` comment. But don't
   create fake data to "demo" the pipeline.

3. **Document assumptions in code.** Every place you make a choice about
   accumulation windows, grid alignment, unit conversions, etc., put a
   comment explaining what you assumed and why.

4. **Ask before guessing.** If you don't know the exact AIFS output
   variable name for precipitation in Earth2Studio, ASK ME or check
   their docs. Don't guess a name and have it silently fail.

5. **Don't add features I didn't ask for.** No web dashboards. No fancy
   CLI argument parsing. No Pydantic validation everywhere. No
   parallelism abstractions. Just the straightforward pipeline.

6. **Use existing libraries.** Don't reimplement regridding,
   cartographic projection, or CRPS computation. Use xesmf, cartopy,
   scoringrules, etc.

7. **Make it reproducible.** Same input → same output. Random seeds
   where applicable. Pinned dependency versions.

## What I'll do separately

- Register for CDS, Earthdata, and IMD access (already in progress)
- CDS is done - the .cdsapirc has my credentials 
- Provision the RunPod H100 instance when scripts/04 is ready to run
- Apply for IMD gridded rainfall access (in parallel; we'll add it later)
- Review your code at each milestone

## Deliverable for this first session

I'd like to come out of this session with:
1. The project skeleton (pyproject.toml, directory structure, README)
2. `scripts/01_test_credentials.py` working end-to-end (assuming I provide credentials)
3. Clear stubs for the rest of the modules with docstrings explaining
   what each will do
4. A list of questions / decisions you need from me before continuing

Please start by reading this prompt back to me at a high level
(3-4 sentences) to confirm you understand the scope, then ask any
clarifying questions before writing code.
