# Monsoon Bias Prototype — Session Status

Snapshot as of 2026-05-28.

## What's working end-to-end

| Script | Status | What it does |
|---|---|---|
| `scripts/01_test_credentials.py` | runs on laptop | Downloads 1 day of ERA5 t2m + 1 day IMERG daily, plots on India map |
| `scripts/02_download_one_date.py` | **green end-to-end** (cache-aware: needs GPU only if forecast NetCDF absent) | Full one-date pipeline: IMERG half-hourly → mm/day, ERA5 hourly tp → mm/day, AIFS 96-h → 4 × tp06 → mm/day, conservative regrid, 3-panel + bias plot |
| `scripts/03_download_all.py` | implemented; not yet exercised at full scale | Batch IMERG + ERA5 for the season, populates the Zarr store, resumable |
| `scripts/04_run_forecasts.py` | **16/122 done** (1 + 3 smoke + 12 batch); `--limit N` for time-boxed sessions | 122 AIFS forecasts with per-forecast trim built in (7.2 GB → 37 MB) |
| `scripts/04b_ingest_aifs.py` | runs on laptop | Idempotent sweep: trimmed AIFS NetCDFs → IMD-day → regrid → Zarr `aifs` var |
| `scripts/05_compute_bias.py` | runs on laptop | 8 NetCDFs of bias diagnostics under `outputs/bias/` |
| `scripts/06_make_plots.py` | runs on laptop | 7 publication PNGs (300 DPI) under `outputs/figures/` |
| `scripts/test_trim.py` | one-shot validator | Bytes-compares trimmed vs untrimmed accumulator output |

All `src/monsoon_bias/` modules either implemented or have detailed docstring stubs:

- **Implemented:** `config.py`, `data/_earthdata.py`, `data/imerg.py`, `data/era5.py`, `data/elevation.py`, `processing/accumulate.py`, `processing/regrid.py`, `processing/store.py`, `forecast/run_aifs.py`, `forecast/trim.py`, `analysis/bias.py`, `analysis/plots.py`
- **Stubbed:** `data/bsiso.py` (deferred — no live index source)

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
3. **Include `earth2studio[aifs]` in the gpu extra *and* pin `anemoi-inference` separately.** `earth2studio[aifs]>=0.14` pulls `anemoi-models`, `earthkit-regrid`, and `ecmwf-opendata` — but *not* `anemoi-inference`, which AIFS actually needs to run. Hit on 2026-05-27 resume. The `[gpu]` extra now pins `anemoi-inference>=0.11` explicitly.
4. **`uv` resolved `earth2studio==0.9.0` for us, not the latest (0.14.0).** The latter wants `numpy 2.x` + `zarr 3.x` + `pandas` downgrade, conflicting with our pins. v0.9.0's CDS lexicon is missing 23 of AIFS's 94 inputs (all `w*` pressure-level vars + 10 surface/soil); v0.14.0's lexicon has them. We patch v0.9.0's lexicon at runtime via `src/monsoon_bias/forecast/_lexicon_patch.py` — clean module copying the 22 lookup-strings verbatim from v0.14.0's source plus one derived entry for `zsl` (AIFS's name for surface geopotential). **Fix for next bootstrap:** either pin `earth2studio==0.14.0` (and accept the dep upheaval, but watch zarr 3 vs our store code), or keep the patch.
5. **Use `set -eu` *without* `pipefail` in bash scripts that include `command | head`.** `nvidia-smi | head -15` triggers SIGPIPE; `pipefail` kills the script. Already fixed in `scripts/00_gpu_setup.sh`.
6. **Anchor rsync excludes with a leading slash** to avoid matching `data` at any depth. `--exclude='data'` will *also* nuke `src/monsoon_bias/data/` along with the top-level `data/` dir. Use `--exclude='/data'` (anchored).
7. **`pgrep -f "<short string>"` matches its own enclosing bash subshell** if the subshell's command line contains the same string. For PID polling, use `kill -0 $PID` with a saved PID file, not `pgrep -f`.
8. **`NetCDF4Backend` creates the output file before the run starts** — if the run crashes mid-fetch, you're left with an empty ~240-byte skeleton that subsequent attempts will silently re-use. `run_aifs.py` now refuses files smaller than 1 MB (size sanity check).
9. **Earth2Studio's `ARCO` data source has a hardcoded `< 2023-11-10` cutoff** in `_validate_time` (their snapshot freeze), even though the live `gcp-public-data-arco-era5` bucket has data well past it (verified to 2026-04). `_patch_arco_date_cutoff()` in `run_aifs.py` lifts the check — but ARCO turned out to also be missing lexicon entries for surface/soil vars, so we use CDS anyway. Patch retained in case we want ARCO later for non-AIFS models.
10. **Earth2Studio caches ARCO chunks to `~/.cache/earth2studio` and never evicts.** On RunPod the container `/` is small (30 GB); each forecast pulls ~5 GB of pressure-level chunks. The cache filled the container disk after ~23 consecutive forecasts on 2026-05-28, killing the batch with `OSError: [Errno 28] No space left on device`. **Fix before any multi-day batch:** symlink the cache to the volume before launching:
    ```bash
    rm -rf /root/.cache/earth2studio
    mkdir -p /workspace/proto-1/.cache/earth2studio
    ln -s /workspace/proto-1/.cache/earth2studio /root/.cache/earth2studio
    ```
    The volume (148 TB) absorbs the cache indefinitely. Also: the empty NetCDF4Backend "skeleton" files (239 B / 48 B) left by ENOSPC-aborted runs need explicit deletion before resume — script 04's `if not path.exists()` skip would otherwise treat them as complete.

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

## Session log — 2026-05-25 (CPU pod, B2 fixed, trim built)

GPU was unavailable on resume; ran on a CPU pod for the day. **B2 is green.**

### What got done

1. **Inspected AIFS NetCDF layout.** Earth2Studio 0.14 emits `(time=[init], lead_time=[0, 6h, …, 96h], lat, lon)` with `tp06` per step. Init is in `time`, not a per-step absolute valid timestamp.
2. **Fixed `accumulate_aifs_to_imd_day`** (`src/monsoon_bias/processing/accumulate.py`) to handle both layouts: new `lead_time`-aware path (selects leads 78/84/90/96 h after asserting init matches), with the old absolute-valid-time path kept as fallback for older serializations.
3. **Made script 02 cache-aware** (`scripts/02_download_one_date.py`): if `data/forecasts/aifs_<init>_nsteps16.nc` already exists and is ≥1 MB, use it without requiring CUDA. Lets B2 run end-to-end on a CPU pod.
4. **B2 verified.** Script 02 produced `outputs/figures/02_one_date_2025-07-15.png`. IMERG 7.48 / ERA5 7.76 / AIFS 8.88 mm/day mean over India. Bias maps show the expected geographically-coherent structure — AIFS dry over central/eastern India, wet over NW Pakistan and the western Himalayan foothills.
5. **Built `forecast/trim.py`.** `trim_aifs_forecast(in_path, out_path=None, delete_source=False)` keeps only `tp06`, zlib level 4, atomic temp-file + `os.replace`, validates round-trip before commit. Sanity guards: trimmed size must be in [1 MB, 500 MB].
6. **Validated trim** with `scripts/test_trim.py` against the real 6.8 GB cached forecast: trimmed to **36 MB (193× smaller)**, IMD-day totals are bytes-identical. **122 forecasts → ~4.5 GB instead of ~880 GB.**
7. **Integrated trim into script 04** batch loop. Trim failures don't abort the batch — they log "TRIM-FAIL" and leave the untrimmed file in place for a later sweep.

### Still untested (low risk)

- **In-place trim path** (`out_path == in_path`, atomic replace). Side-file path is bytes-validated; the in-place path adds only `os.replace`. Verify on resume by trimming a copy in-place before launching the 122-batch.

### Pod state at end of session (stopped)

```
data/forecasts/
  aifs_20250712T0300_nsteps16.nc          6.8 GB  (original, kept for in-place trim test)
  aifs_20250712T0300_nsteps16.trimmed.nc   36 MB  (side-file trim, validated)
```

## Session log — 2026-05-27 (H100 resumed; B3 partial; IMERG/ERA5 confirmed populated)

### What got done

1. **Resumed on H100 SXM** in US-MO-1 (RunPod Pytorch 2.8.0 template, `fair_lavender_asp_volume` attached at `/workspace`). Env survived from prior pod stop — torch 2.8.0+cu128, flash_attn 2.8.3, earth2studio 0.14.0 all intact.
2. **Discovered missing dep: `anemoi-inference` is NOT pulled by `earth2studio[aifs]>=0.14`.** Manually installed (0.11.0) and pinned in `pyproject.toml` `[gpu]` extra so the next bootstrap doesn't repeat the find.
3. **In-place trim verified.** Trimmed the cached 7.20 GB `aifs_20250712T0300_nsteps16.nc` in-place → 37.4 MB (193×). IMD-day totals byte-identical to the pre-existing side-file `.trimmed.nc` reference (max abs diff = 0.0 mm/day). The remaining untested item from 2026-05-25 is now green.
4. **B2 regression check passed.** `scripts/02_download_one_date.py` with the trimmed forecast produces identical numbers: IMERG 7.48 / ERA5 7.76 / **AIFS 8.88 mm/day mean over India**, matching 2026-05-25's run exactly.
5. **`scripts/04 --limit N`.** Added a `--limit` flag so time-boxed GPU sessions can run a subset; resumability already handles continuation.
6. **Smoke test (3 forecasts, --limit 3):** 3/3 succeeded in 8.5 min wall clock.
7. **B3 batch (12 forecasts, --limit 12):** 12/12 succeeded in 35.8 min wall clock.
8. **Confirmed local Zarr already has 121/122 days of IMERG + ERA5** populated from a prior session that wasn't captured in STATUS.md. Only 2025-09-30 is missing (the known IMERG-not-fully-published issue). Script 03 effectively done.

### Per-forecast wall clock (network-volume H100 SXM)

Variance is real: cold-start #1 took 5.4 min (process boot + CUDA kernel warmup + ARCO cold cache). Steady-state spread from 1.8 min → 4.2 min depending on ARCO chunk latency + network volume I/O. **Mean ~2.5–3 min** rather than the optimistic ~2 min STATUS.md had been quoting. Re-estimating the full 122 batch at this rate: ~6 hr (not 4) on H100 SXM.

### Pod state at end of session (stopped)

```
data/forecasts/                       16 AIFS NetCDFs trimmed to ~37 MB each (~590 MB total)
  aifs_20250712T0300_nsteps16.nc                       (in-place trimmed)
  aifs_20250712T0300_nsteps16.trimmed.nc               (side-file, original validation reference)
  aifs_20250712T0300_nsteps16.trimmed.nc.refbackup     (preserved during in-place test; can delete)
  aifs_20250529T0300..20250612T0300_nsteps16.nc       (15 fresh from smoke + batch)
```

`/workspace/proto-1` on the pod is still not a git checkout — it's the original rsync tree. We're patching it via `scp` for the rare local edits we want to push to the pod. Cleanup item.

### Cost this session: ~$2.50 (38 min of H100 SXM at $3.29/hr + storage). Cumulative: ~$14.50.

## Session log — 2026-05-28 (full pipeline online; B3 batch running)

### What got done

1. **Resumed on H100 SXM** (new port, same volume in US-MO-1, RunPod Pytorch 2.8.0 template). Env fully intact: torch 2.8.0+cu128, flash_attn 2.8.3, earth2studio 0.14.0, anemoi.inference 0.11.0 (the pin from 2026-05-27 stuck).
2. **Launched the full 106-forecast batch in the background** via `scripts/04_run_forecasts.py` (no `--limit`). nohup'd so it survives SSH disconnects. Per-forecast wall clock running ~2-3.5 min, on pace for ~5.5 hr total.
3. **Wrote `scripts/04b_ingest_aifs.py`** — idempotent sweep that reads trimmed AIFS NetCDFs from `data/forecasts/`, accumulates each to an IMD-day total, regrids to the common 0.25° grid, and writes the `aifs` variable into the master Zarr store. Validates against script 02 output (2025-07-15 mean=8.88 mm/day, byte-identical).
4. **scp'd 21 trimmed AIFS NetCDFs from pod to laptop** (filter: <50 MB, idle ≥2 min, to skip in-progress 7 GB writes). Ran the sweep → **21/122 AIFS days populated** in the local Zarr alongside the existing 121 IMERG + 121 ERA5.
5. **Implemented `analysis/bias.py`** — five non-deferred diagnostics: `mean_bias_map`, `rmse_map`, `bias_by_region`, `bias_by_elevation`, `bias_by_rainfall_magnitude`. Each accepts a forecast/observed pair so the same code computes AIFS−IMERG, ERA5−IMERG, and the AIFS−ERA5 residual.
6. **Verified `data/elevation.py` was already implemented** (STATUS.md previously misclassified as stub). NOAA ETOPO 2022 60-arc-sec → regridded to 0.25°. Mumbai 23 m, Delhi 222 m, Himalaya (30N, 80E) 1987 m — all sensible.
7. **Implemented `analysis/plots.py`** — seven plot functions: bias map (`cmocean.balance`), RMSE map (`cmocean.amp`), three-panel decomposition, region bar chart, elevation-bin bar chart, bias-vs-elevation hexbin scatter, rainfall-magnitude bar chart. BSISO panel remains deferred.
8. **Wrote `scripts/05_compute_bias.py` + `scripts/06_make_plots.py`** as thin orchestrators. End-to-end smoke against the 21-day cube produces 8 NetCDFs under `outputs/bias/` and 7 PNGs under `outputs/figures/`.

### What the partial cube already shows

Three diagnostic findings from just 21 days (early-monsoon, Jun 1–20 + the 07-15 cached date):

1. **Rainfall-magnitude tail miscalibration (the smoking gun).** AIFS over-predicts trace (+1.99 mm/day) and light (+3.52) rain, then under-predicts moderate (−4.48), heavy (−27.47), and very-heavy (−75.44 mm/day). Sample count remains substantial in the heavy tail (n=12,398 / n=4,080). This is the canonical AI-weather-model failure: a narrower precip distribution than reality.
2. **Orographic bias.** Western Ghats windward face dry (−1.34), leeward wet (+1.27); Himalayan foothills wet (+1.73); plains slightly dry (−1.75); higher elevations consistently wet (+1.36 to +2.50). ERA5 shows the same shape with smaller magnitude.
3. **AIFS−ERA5 residual has its own structure** (3-panel figure, right panel). Not random — AIFS departs from its training distribution in spatially structured ways.

All three patterns are exactly the kind of structured, learnable bias the prototype was set up to test for. This validates the project's precondition: **bias has learnable structure → a corrective head is justified.**

### Pod state during this session

```
data/forecasts/      ~16+(N batch-completed) AIFS NetCDFs, all trimmed to ~37 MB each
logs/b3_full_*.log   batch progress log (per-forecast lines)
/tmp/b3.pid          batch PID
```

`/workspace/proto-1` is still not a git checkout; we sync code edits with `scp`. Cleanup item.

### Cost this session: ~$18 estimated when batch finishes (~5.5 hr × $3.29/hr). Cumulative ~$32.

## Resume checklist (paused 2026-05-28 PM, pod stopped)

### Where things were when the session ended

- Batch was alive on the pod at `[N/106]` (check `tail logs/b3_full_*.log` to find the exact stopping point). Resume with `scripts/04_run_forecasts.py` (no `--limit`) — it skips already-completed forecasts.
- Local Zarr had 121 IMERG + 121 ERA5 + (21 + however many batch completions were synced) AIFS days.

### Next steps

1. **Resume pod**, verify env probe, kick off `scripts/04_run_forecasts.py` again to finish whatever wasn't done last time.
2. **scp the new trimmed AIFS NetCDFs** to the laptop:
   ```bash
   ssh ... 'find /workspace/proto-1/data/forecasts -name "aifs_*.nc" -size -50M -not -name "*.trimmed.nc" -printf "%f\n"' \
     | rsync --files-from=- ... root@...:/workspace/proto-1/data/forecasts/ data/forecasts/
   ```
3. **`uv run python scripts/04b_ingest_aifs.py`** — idempotent, populates new days into the local Zarr.
4. **`uv run python scripts/05_compute_bias.py && uv run python scripts/06_make_plots.py`** — regenerates the 8 NetCDFs + 7 PNGs against the now-fuller cube.
5. **Compare full-122 figures to the 21-day partial snapshot** in `outputs/figures/`. The three structural findings (rainfall-magnitude tail, orography, residual) should sharpen, not invert; if any flip sign, it's worth investigating.

### Long-tail / nice-to-have

- **Land mask.** Currently no mask applied; plots include ocean cells. Cheapest fix: use `cartopy.feature.OCEAN` or compute a binary from ETOPO ≥ 0 m to mask out bathymetry-clipped ocean. Add as a `bias.land_mask(grid) → DataArray[bool]` helper.
- **Pod git checkout.** `/workspace/proto-1` is still a rsync tree. Convert via `git init && git remote add origin ... && git fetch && git checkout origin/main -- .` (destructive — overwrites tracked files, untracked data/ and .venv survive). Lets us `git pull` instead of scp.
- **BSISO stratification.** Self-compute from NOAA OLR + IPRC EEOF vectors per Kikuchi 2012. ~150 LoC. The plotting layer (`plot_bias_by_bsiso`) is stubbed and ready.

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
