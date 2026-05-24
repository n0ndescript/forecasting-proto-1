# Monsoon Bias Prototype — Session Status

Snapshot as of 2026-05-21.

## What's working end-to-end

| Script | Status | What it does |
|---|---|---|
| `scripts/01_test_credentials.py` | runs on laptop | Downloads 1 day of ERA5 t2m + 1 day IMERG daily, plots on India map |
| `scripts/02_download_one_date.py` | runs on laptop *except* AIFS step | Full one-date pipeline: IMERG half-hourly → mm/day, ERA5 hourly tp → mm/day, conservative regrid, bias plot |
| `scripts/03-06` | stubs | Not yet implemented |

All `src/monsoon_bias/` modules either implemented or have detailed docstring stubs:

- **Implemented:** `config.py`, `data/_earthdata.py`, `data/imerg.py`, `data/era5.py`, `processing/accumulate.py`, `processing/regrid.py`, `forecast/run_aifs.py` (untestable without GPU but written against confirmed Earth2Studio 0.14.0 API)
- **Stubbed:** `data/bsiso.py`, `data/elevation.py`, `processing/store.py`, `analysis/bias.py`, `analysis/plots.py`

## Concrete results

**Script 01** for 2025-07-15:

- ERA5 t2m and IMERG daily downloaded, plotted, look meteorologically correct.

**Script 02** for the same date (laptop, no AIFS):

- 48 IMERG half-hourly granules downloaded and summed → IMD-day mm/day (mean 7.48, max 135.9 mm/day over India)
- 48 hours of ERA5 hourly `tp` downloaded, sliced to the 24 stamps in the IMD window → mean 7.76, max 201.8 mm/day
- Both conservatively regridded to a common 0.25° India grid (129 × 121)
- ERA5 − IMERG bias map shows **clear geographically-coherent structure**: wet bias over central India and Himalayan foothills, dry bias on the Western Ghats windward coast. This is the *kind* of structure the project hypothesis predicts, but for ERA5; AIFS will be added once GPU is available.

## Decisions made (with rationale + tradeoffs)

### Hard infrastructure choices

| Decision | Rationale | Tradeoff accepted |
|---|---|---|
| **`uv` for dep mgmt** | User preference; faster than pip | — |
| **CPU base + `[gpu]` extra** | Laptop can run analysis/regrid/plot; H100 only needed for AIFS | Separate install step on GPU host |
| **xarray-regrid instead of xesmf** | Pure Python, no system deps; ESMF not in brew core on macOS | Loses unstructured/non-rectilinear grid support (don't need it — IMERG, ERA5, AIFS are all rectilinear lat/lon) |
| **EDL token via `.edl_token` file, bypassing `earthaccess.login()`** | URS's `find_or_create_token` endpoint rejects newly-created accounts; web login works fine. Direct CMR search + Bearer-token GES DISC downloads work universally | Token must be regenerated every 60 days |

### Scientific / data choices

| Decision | Rationale | Tradeoff |
|---|---|---|
| **AIFS as primary forecast model** | Operational ECMWF AI model, has native precip output (`tp06`); GraphCast needs separate diagnostic models for precip | If AIFS turns out to be biased uniformly, won't know whether GraphCast is too |
| **AIFS-Single 1.1.0** specifically | Released 2025-08-27; fixes the negative-precipitation bug present in 1.0.0. Earth2Studio 0.14.0 ships 1.1.0 by default | Accumulator clips negatives to 0 defensively and warns if >0.1% of cells are negative — surfaces a stale package cache |
| **ARCO-ERA5 (GCS) as AIFS initial-condition source** | (a) ARCO is hourly and unrestricted, so 03 UTC inits work for any date in the ERA5 archive; (b) ARCO *is* ERA5 — matches AIFS's training distribution exactly (in-distribution ICs); (c) no CDS auth, no rate limits, no queue. The Earth2Studio default `IFS_FX` source only serves 00/06/12/18 UTC ICs for the last 4 days — disqualified on both axes. `CDS` would work but is slow, rate-limited, and adds auth surface | None — strictly better than the alternatives for our use case |
| **IMERG Final V07** (half-hourly, `GPM_3IMERGHH`) | Highest-quality satellite obs; 3.5-month latency irrelevant for 2025 monsoon | 48 granules/day → ~70 MB/day download, ~8 GB for full season |
| **2025 monsoon (Jun 1 – Sep 30)** instead of 2024 | Same statistical content; more recent; both IMERG and ERA5 are published past their latency window for this period as of 2026-05-19 | None meaningful |
| **Include ERA5 baseline panel** | Enables 3-way decomposition: AIFS−IMERG / ERA5−IMERG / AIFS−ERA5. Cleanly separates model-specific bias from training-data/observation mismatch | +50% CDS calls, +1 variable in Zarr, plots become 3-up rows |
| **Daily forecasts (122/season)** instead of every-3rd-day | Cost trivial (~$6 on H100), preserves statistical power for BSISO phase stratification (~15 dates/phase vs ~5) | None — sub-sampling only made sense if cost were an issue |
| **AIFS init at 03 UTC**, *not* the standard 00 UTC | Aligns the 6-h AIFS steps with the IMD-day boundary (03→03 UTC). Lets us sum 4 raw `tp06` values to get one IMD-day total — no time interpolation, which would break accumulation semantics | Uncommon init hour; no direct comparison against ECMWF operational forecasts (which init at 00/12 UTC) |
| **96-h forecast (nsteps=16)**, not 72 | The 3-day-lead verifying day starts 72 h after init and ends at 96 h. A 72-h forecast wouldn't cover the verifying day | +33% GPU time per forecast (already cheap) |
| **Conservative regridding only for precip** | Bilinear doesn't preserve total water; would silently corrupt the bias | Module refuses bilinear on precip (`regrid_precip` validates units = mm/day before running) |

### Analysis-design choices

| Decision | Rationale | Tradeoff |
|---|---|---|
| ~~**BSISO phase stratification**~~ **DEFERRED** (see Known limitations) | Originally planned via Kikuchi index | Kikuchi data source is dormant; self-compute path exists but deferred to a follow-up |
| **6 rectangular region bboxes** (Western Ghats windward/leeward, Gangetic, NE, foothills, peninsular interior) | Captures the meteorologically meaningful sub-regions; quick to set up | Crude; would be better with IMD subdivision shapefiles. Easy to swap later |
| **ETOPO1 (1 arc-min) for elevation** | One file, no tiling; coarsens to 0.25° cleanly | SRTM 30m would be overkill — we average into 0.25° cells anyway |
| **Bias = forecast − observed** convention | Standard sign convention; positive = wet bias in the model | — |
| **Land-only stratification masks** | IMERG retrieval quality differs over ocean | Means we'll need a land mask — placeholder in `analysis/bias.py` |

## Lessons for the next GPU bootstrap

When provisioning a new GPU host (RunPod or otherwise), these things bit us; fix in `pyproject.toml` next time:

1. **Pin torch to a version that has pre-built `flash_attn` wheels.** Our `torch>=2.2` constraint let uv pull the latest (2.12.0 with CUDA-13 wheels), which mismatched the pod's CUDA-12.8 toolkit. Reinstalling with `--index-url https://download.pytorch.org/whl/cu128` brought it to 2.11.0+cu128 — but `flash_attn` doesn't publish pre-built wheels for torch 2.11+cu128, so we had to compile from source (~30 min on H100 across 4 GPU archs, ~10 min restricted to sm_90 only). For this prototype we downgraded to **torch 2.8.0+cu128** and grabbed the matching pre-built wheel from <https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3> (instant). **Fix in pyproject.toml:** narrow to `torch>=2.7,<=2.8` and add the wheel index in install instructions.
2. **Set `FLASH_ATTN_CUDA_ARCHS=90` (or target arch) before any source build.** Default compiles for sm_80, sm_90, sm_100, sm_120 — 4× the work for nothing.
3. **Include `earth2studio[aifs]` in the gpu extra.** Plain `earth2studio` doesn't pull `anemoi.inference`, `anemoi.models`, `earthkit.regrid`, `ecmwf.opendata` — AIFS won't import without those.
4. **`uv` resolved `earth2studio==0.9.0` for us, not the latest (0.14.0).** The latter wants `numpy 2.x` + `zarr 3.x` + `pandas` downgrade, conflicting with our pins. v0.9.0's CDS lexicon is missing 23 of AIFS's 94 inputs (all `w*` pressure-level vars + 10 surface/soil); v0.14.0's lexicon has them. We patch v0.9.0's lexicon at runtime via `src/monsoon_bias/forecast/_lexicon_patch.py` — clean module copying the 22 lookup-strings verbatim from v0.14.0's source plus one derived entry for `zsl` (AIFS's name for surface geopotential). **Fix for next bootstrap:** either pin `earth2studio==0.14.0` (and accept the dep upheaval, but watch zarr 3 vs our store code), or keep the patch.
5. **Use `set -eu` *without* `pipefail` in bash scripts that include `command | head`.** `nvidia-smi | head -15` triggers SIGPIPE; `pipefail` kills the script. Already fixed in `scripts/00_gpu_setup.sh`.
6. **Anchor rsync excludes with a leading slash** to avoid matching `data` at any depth. `--exclude='data'` will *also* nuke `src/monsoon_bias/data/` along with the top-level `data/` dir. Use `--exclude='/data'` (anchored).
7. **`pgrep -f "<short string>"` matches its own enclosing bash subshell** if the subshell's command line contains the same string. For PID polling, use `kill -0 $PID` with a saved PID file, not `pgrep -f`.
8. **`NetCDF4Backend` creates the output file before the run starts** — if the run crashes mid-fetch, you're left with an empty ~240-byte skeleton that subsequent attempts will silently re-use. `run_aifs.py` now refuses files smaller than 1 MB (size sanity check).
9. **Earth2Studio's `ARCO` data source has a hardcoded `< 2023-11-10` cutoff** in `_validate_time` (their snapshot freeze), even though the live `gcp-public-data-arco-era5` bucket has data well past it (verified to 2026-04). `_patch_arco_date_cutoff()` in `run_aifs.py` lifts the check — but ARCO turned out to also be missing lexicon entries for surface/soil vars, so we use CDS anyway. Patch retained in case we want ARCO later for non-AIFS models.

## Workarounds we navigated

1. **CDS Beta API.** New format requires `data_format`/`download_format` keys and the new endpoint URL. Worked first try once `.cdsapirc` was correctly configured.
2. **macOS zsh history expansion ate the `!` in the Earthdata password** when writing `~/.netrc`. Fixed by writing the file with a single-quoted heredoc.
3. **URS `find_or_create_token` endpoint rejects newly-created accounts even when the web login works.** Bypassed by generating an EDL token manually in the URS profile and using it directly in `Authorization: Bearer …` headers against CMR + GES DISC. This may also be the right pattern long-term — tokens are simpler than netrc.
4. **ESMF not available in homebrew core, esmpy not pip-installable cleanly on macOS arm64.** Swapped to `xarray-regrid`, which is pure Python and supports conservative regridding for rectilinear grids — exactly our case.
5. **IMERG times are stored as `cftime.DatetimeJulian`, not numpy datetime64.** Convert via `isoformat()` round-trip before comparing to pandas Timestamps.
6. **`requests` strips `Authorization` header on cross-host redirects** (URS → GES DISC). Manual redirect-following loop in `_earthdata.download_with_token` re-attaches the header on each hop.
7. **AIFS 6-h native step is offset from the IMD-day boundary** if you init at 00 UTC. Solved by initializing at 03 UTC (covered in detail above).

## Known limitations (intentionally accepted)

### 2025-09-30 missing from the dataset (121/122 days)

- The IMD-day window for 2025-09-30 ends 2025-10-01 03 UTC, which is slightly past where IMERG Final V07 had fully published at the time of download (2026-05-21).
- CMR returned 42 of the expected 48 half-hourly granules; pipeline correctly refused to accept partial data.
- **Impact:** ~0.8% of the season missing. Negligible for mean-bias / RMSE diagnostics.
- **Fix path if needed:** re-run `scripts/03_download_all.py` in a few weeks (the remaining 6 granules should publish), or substitute IMERG Late Run for that single day.

### BSISO phase stratification deferred

- **Originally planned:** stratify bias maps by the Kikuchi BSISO phase (1–8) to test whether bias has *temporal* structure tied to monsoon active/break regimes.
- **Why deferred:** the Kikuchi IPRC `rt_pc.txt` file last updated 2022-12-29 — useless for the 2025 monsoon. APCC's Lee et al. realtime product (the obvious alternative) blocks direct URL access and would require an API arrangement.
- **What we lose:** the BSISO row in the analysis (~1 of 6 stratifications). The headline bias map, RMSE map, elevation/region/rainfall-magnitude stratifications all still run.
- **What it would take to add:** self-compute the Kikuchi index from NOAA Interpolated OLR + the EEOF vector files that *are* still hosted at IPRC. ~150 lines of code; methodology in Kikuchi, Wang & Kajikawa 2012. Recipe is documented in `src/monsoon_bias/data/bsiso.py` for the future pass.

## Verification checklist for the first GPU run

Before launching the 122-day batch, run script 02 once on the H100 and
confirm — these are the things doc-reading can't tell you:

1. **Variable coverage.** Earth2Studio will fail fast on the first AIFS
   call if ARCO is missing any of the ~89 input variables AIFS needs.
   If it does, supplement from CDS or check whether a different ARCO
   collection has them.
2. **`tp06` semantics.** Open the output NetCDF. Confirm: (a) units
   string is `m` (or `mm`); (b) values are *per-step accumulation*, not
   a running total — sum across all 16 steps should be the 96-h total;
   (c) negative-cell fraction is well below 0.1 % (accumulator will warn
   if not — indicates a stale AIFS 1.0.0 package).
3. **Lead-time indexing.** `run([t0], 16, ...)` should produce 16 output
   steps with times `t0 + 6h, t0 + 12h, ..., t0 + 96h`. Step 0 (the IC
   itself) may or may not be included — confirm before slicing for the
   IMD day.
4. **Spatial grid.** Output should be on a regular 0.25° lat/lon grid
   (Earth2Studio regrids AIFS's internal O96 reduced Gaussian). Our
   accumulator + regridder assume this; verify shape and coord values.

## Resume checklist (paused 2026-05-22 PM, pod stopped)

### Where things stand at the end of this session

- **Pod environment: working** with earth2studio 0.14.0 + torch 2.8.0+cu128 + flash_attn 2.8.3. Resolved during the session:
  - Pulled `nvidia-cusparselt-cu12` + correct `nvidia-nccl-cu12` via an atomic `uv pip install --reinstall torch==2.8.0 --index-url ...cu128` (uv sync alone left an inconsistent CUDA stack).
  - Added `[tool.uv]` block to `pyproject.toml` for `no-build-isolation-package = ["flash-attn", ...]` and `[tool.uv.extra-build-dependencies]` so flash-attn doesn't get rebuilt from source on every sync.
- **ARCO path verified end-to-end through inference.** 90 inputs fetched in ~40 s, 17 steps inferred in ~50 s on H100 SXM, 7.2 GB NetCDF written.
- **`_lexicon_patch.py` still required for ARCO**: v0.14.0's ARCO lexicon is missing 5 surface vars (`tcw, swvl1, swvl2, stl1, stl2`). Patch is now re-enabled in `run_aifs.py`.
- **Accumulator bug fixed in source**: now accepts empty units attribute (AIFS v0.14's `tp06` has no `units` attr; ECMWF convention is meters).
- **B2 *still not green*** — one remaining accumulator issue:
  - `ValueError: AIFS output missing steps [..09:00, ..15:00, ..21:00, ..+1d 03:00] required ... Available: [Timestamp('2025-07-12 03:00:00')]`
  - AIFS NetCDF on pod has all 16 forecast steps' worth of data (7.2 GB), but the accumulator is reading them under the wrong coordinate. Earth2Studio 0.14's `NetCDF4Backend` likely stores forecast steps along a `lead_time` dim (or `time` is the init only) — accumulator currently looks for absolute valid-time stamps in `time`.

### Next session — first task

1. **Resume pod**, get new SSH info.
2. **Inspect the AIFS NetCDF dim structure** (~30 s):
   ```bash
   ssh ... "cd /workspace/proto-1 && uv run python -c '
   import xarray as xr
   ds = xr.open_dataset(\"data/forecasts/aifs_20250712T0300_nsteps16.nc\")
   print(dict(ds.dims)); print(list(ds.coords))
   for c in ds.coords: print(c, ds[c].shape, ds[c].values[:5] if ds[c].size <= 20 else ds[c].values[:3])
   print(ds.tp06.dims, ds.tp06.shape)'"
   ```
3. **Update `accumulate_aifs_to_imd_day`** to use whatever coord Earth2Studio actually emits (likely `lead_time` + `init_time`, with valid-time computed as their sum). Should be ~10 LoC change.
4. **Re-run B2** — no GPU work, just imports + accumulator + plot (~3 min). Should produce the 3-panel `outputs/figures/02_one_date_2025-07-15.png` with AIFS + ERA5 + IMERG.

### After B2 passes — open issues for B3 (the 122-forecast batch)

- **Disk:** each forecast NetCDF is ~7.2 GB (AIFS dumps every output variable). 122 forecasts × 7.2 GB = 880 GB; pod disk is 80 GB. Need per-forecast trimming: after each AIFS run, slice out only `tp06`, save ~80 MB, delete the 7.2 GB original.
- **Per-forecast wall clock:** ~50 s ARCO + ~50 s inference + ~5 s I/O = ~2 min/forecast. 122 × 2 min ≈ 4 hours. ~$12 GPU at $2.90/hr.

### Cost so far: ~$10 of GPU + ~$1 storage. Pod stopped at session end.

## Stack snapshot

```
Python 3.11
uv 0.11.8
xarray 2026.4.0, pandas 3.0.3, numpy 1.26.4
zarr 2.18.7
xarray-regrid 0.4.2          ← regridding (pure Python, no ESMF)
cartopy 0.22+, cmocean 3.0+   ← plotting
cdsapi 0.7.4+                 ← ERA5 baseline panel
requests                      ← IMERG (via CMR + EDL bearer token)
earth2studio 0.14.0, torch 2.2+   ← GPU only; AIFS ICs from ARCO (no auth)
```
