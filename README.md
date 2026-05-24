# Monsoon Bias Prototype

Diagnostic pipeline for evaluating whether off-the-shelf global AI weather
models (ECMWF AIFS, Google GraphCast) have systematic, geographically-coherent
biases when applied to the Indian monsoon region.

This prototype tests the precondition for a follow-on project: **does the
bias have learnable structure?** It does not build a corrective model.

## Scope

- Target season: **June 1 – September 30, 2025** (Indian monsoon).
- Forecast model: **ECMWF AIFS**, run via NVIDIA Earth2Studio from ERA5
  initial conditions, 3-day lead.
- Ground truth: **NASA GPM IMERG Final V07** (half-hourly, accumulated to
  IMD-day windows of 03:00 UTC → 03:00 UTC the next day).
- Spatial domain: India bbox (lat 6°N–38°N, lon 68°E–98°E) at 0.25°.

## Setup

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Python deps (CPU base)

```bash
uv sync
```

Regridding uses `xarray-regrid` (pure-Python, no system deps). If you
later need non-rectilinear or unstructured-grid support, swap to `xesmf`
— it needs the ESMF C library, install via conda-forge.

### 3. GPU deps (on the RunPod H100, not laptop)

```bash
# Install a CUDA-matched torch wheel first, then:
uv sync --extra gpu
```

### 4. Credentials

**CDS (ERA5):** put your key in `~/.cdsapirc` or the project root:

```
url: https://cds.climate.copernicus.eu/api
key: <your-uid>:<your-api-key>
```

**Earthdata (IMERG):** register at
<https://urs.earthdata.nasa.gov/users/new>, then create `~/.netrc`:

```
machine urs.earthdata.nasa.gov login <user> password <pass>
```

`chmod 600 ~/.netrc`. Also accept the GES DISC EULA at
<https://disc.gsfc.nasa.gov/earthdata-login>.

## Running

```bash
uv run python scripts/01_test_credentials.py    # smoke test
uv run python scripts/02_download_one_date.py   # one date end-to-end
uv run python scripts/03_download_all.py        # 122 days of ERA5+IMERG
uv run python scripts/04_run_forecasts.py       # GPU: run AIFS forecasts
uv run python scripts/05_compute_bias.py        # CPU: bias diagnostics
uv run python scripts/06_make_plots.py          # CPU: figures
```

## Layout

```
src/monsoon_bias/
  config.py             # paths, bboxes, time periods, constants
  data/                 # downloaders (ERA5, IMERG, BSISO, elevation)
  forecast/             # AIFS via Earth2Studio
  processing/           # regridding, accumulation, Zarr I/O
  analysis/             # bias diagnostics + plots
scripts/                # numbered pipeline scripts (01 → 06)
```

## Status

See `context.md` for the full project brief.
