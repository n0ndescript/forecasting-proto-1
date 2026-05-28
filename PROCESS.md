# Prototype 1 — process and methodology

A companion to `WRITEUP.md`. The brief tells you *what we found*; this doc tells
you *what we set out to do, why we made the choices we made, and what we'd do
next*. Written so a future collaborator (or future you) can pick this up
without re-reading the chat log.

---

## 1. The thesis we set out to validate

### One-sentence version

**Does the precipitation bias of off-the-shelf global AI weather models, when
applied to the Indian monsoon, have learnable structure?**

### Why it matters

There is a class of follow-on projects ("corrective heads", region-specific
post-processing models) that *only make sense* if the answer is yes. If global
AI weather models are biased over India in spatially incoherent, regime-free,
unstructured ways, then a small downstream model has nothing to learn from —
there is no structure to extract. Conversely, if the bias is structured, the
follow-on projects are not just defensible but probably high-value.

This question is upstream of every reasonable follow-on. It deserves to be
answered before anyone commits to building one.

### Three claims this prototype was set up to support or refute

1. AIFS bias over India has **spatial structure** — coherent regional patterns
   rather than salt-and-pepper noise.
2. AIFS bias has **rainfall-magnitude structure** — systematic over- or
   under-prediction conditional on how much it was actually raining.
3. The structure is **AIFS-specific**, not just inherited from its training
   data (ERA5). I.e., the AIFS−ERA5 residual is itself structured.

If all three hold, the precondition is met. If none hold, we report a null
result and the follow-on stops here.

### What this prototype was explicitly *not* trying to do

- Build the corrective model itself.
- Compare AIFS to other global AI weather models (GraphCast, Pangu, etc.).
- Evaluate AIFS's skill metric-by-metric — only the bias structure.
- Generalize beyond a single monsoon season.

These are scoped out, not forgotten. They are the obvious follow-ons.

---

## 2. Design decisions, with rationale

This section exists because every choice below has a defensible alternative
that we could be asked to justify.

### Why AIFS

- **Operational ECMWF model.** Has institutional momentum and a roadmap; not a
  research artifact that will disappear in a year.
- **Native `tp06` precipitation output.** GraphCast, by contrast, doesn't
  produce precipitation directly — it requires a separate diagnostic model
  that introduces its own biases. Starting with the cleaner case.
- **AIFS-Single 1.1.0** specifically: released 2025-08-27, fixes the
  negative-precipitation bug in 1.0.0. Earth2Studio 0.14 ships 1.1.0 by default.
- Tradeoff accepted: if AIFS turns out to be uniformly biased, we don't know
  whether GraphCast is too. We accept this in exchange for cleaner setup.

### Why 2025 monsoon, why Jun 1 – Sep 30, why 122 days

- **2025** because both IMERG Final V07 (3.5-month latency) and ERA5 final
  (3-month latency) are fully published for this window as of mid-2026.
- **Jun 1 – Sep 30** is the canonical Indian monsoon. Onset varies (typically
  late May – early June), withdrawal varies (typically late September), but
  this window covers the bulk of the season for every year.
- **Daily verifying dates (122 / season)** rather than every-3rd-day:
  $6 of GPU cost is irrelevant; daily preserves statistical power for any
  follow-on stratification (e.g., BSISO active vs break, if a future analysis
  has access to a working index).

### Why ARCO-ERA5 as the initial-condition source

- **Hourly inits.** The standard Earth2Studio `IFS_FX` source only serves
  00 / 06 / 12 / 18 UTC and only for the last 4 days. We need 03 UTC init for
  every day of the 2025 monsoon — only ARCO has it.
- **No auth, no rate limits, no CDS queue.** Each fetch is ~30 s vs CDS's
  10–30 min.
- **In-distribution.** AIFS was trained on ERA5; ARCO *is* ERA5. We are not
  asking AIFS to extrapolate out of its training distribution at init time.

Patched two things at runtime:

- Earth2Studio 0.14's ARCO lexicon is missing 5 surface vars (`tcw`, `swvl1`,
  `swvl2`, `stl1`, `stl2`). `_lexicon_patch.py` adds them.
- Earth2Studio's ARCO wrapper has a defensive `< 2023-11-10` cutoff in
  `_validate_time` (snapshot freeze of the test data). The live bucket has
  data well past that. `_patch_arco_date_cutoff()` lifts the check.

### Why 03 UTC init, not the standard 00 UTC

- IMD's definition of a "rainfall day" runs 08:30 IST → 08:30 IST (next day),
  which is 03:00 UTC → 03:00 UTC.
- AIFS native step is 6 h, fixed. With a 00 UTC init, valid times land on
  06/12/18/24 UTC and *cannot be summed to align with the IMD window without
  time interpolation* (which would corrupt accumulation semantics).
- With a 03 UTC init, valid times land on 09/15/21/03 UTC, and four
  consecutive 6-h steps sum exactly to one IMD day.
- Tradeoff: 03 UTC is an uncommon init hour, so we cannot directly compare
  our AIFS runs to ECMWF operational output (which inits at 00 / 12 UTC). We
  accept this in exchange for a clean accumulation alignment.

### Why a 96-h (16-step) forecast, not 72

- A 3-day-lead verifying day starts 72 h after the 03 UTC init and ends at
  96 h. A 72-h forecast wouldn't cover the verifying day; it would end exactly
  at the IMD-day boundary.
- We need leads at 78, 84, 90, 96 h. So we run 16 steps. +33% more GPU per
  forecast than 12 steps — irrelevant on H100 wall clock.

### Why conservative regridding only for precipitation

- Bilinear interpolation does not preserve total water. Regridding precip
  bilinearly silently corrupts the bias — the corrupted fields look fine on
  inspection but the conserved quantity is wrong.
- `regrid_precip` refuses anything but conservative and validates that the
  input has units of `mm/day` before running.
- `regrid_continuous` exists for non-conserved fields (2 m temperature etc.)
  and explicitly rejects use on precip-typed fields.

### Why xarray-regrid (not xesmf)

- xesmf depends on the ESMF C library, which is not in homebrew core on
  macOS arm64 and is painful to install.
- xarray-regrid is pure Python, supports conservative regridding for
  rectilinear grids, and our source grids (IMERG, ERA5, AIFS post-Earth2Studio
  regrid) are all rectilinear.
- Tradeoff: we lose unstructured-grid / non-rectilinear support. We accept
  this — none of our data is on such a grid.

### Why ERA5 as a baseline, not just IMERG

- Enables a three-way decomposition: AIFS − IMERG, ERA5 − IMERG, AIFS − ERA5.
- The third quantity is the AIFS-specific residual. Without it, we cannot
  tell whether AIFS bias is just inherited from ERA5 (its training data) or
  is novel error AIFS adds on top.
- Cost: +50 % CDS calls, +1 variable in the Zarr master store, plots get a
  third panel. Acceptable.

### Why a master Zarr store with prealloc-NaN, not per-day NetCDFs

- We know all 122 dates upfront. Preallocating a NaN-filled `(122, lat, lon)`
  array per variable means per-day writes are single-chunk region writes
  (`to_zarr(region={"time": slice(i, i+1)})`), which is O(1) instead of
  O(season).
- The downstream analysis ("mean across time", "groupby rainfall bin") is
  trivially efficient on a single Zarr cube; would require a manifest +
  open-with-mfdataset on per-day NetCDFs.
- Compressed size of the fully populated cube: ~5 MB. The "manifest"
  argument doesn't apply at this scale, but the indexing argument does.

---

## 3. Method, end-to-end

The pipeline is six numbered scripts plus a sweep:

```
scripts/01_test_credentials.py      smoke test for CDS + Earthdata auth
scripts/02_download_one_date.py     full pipeline for one date end-to-end
scripts/03_download_all.py          IMERG + ERA5 batch for the season, → Zarr
scripts/04_run_forecasts.py         122 AIFS forecasts on GPU, trim each in place
scripts/04b_ingest_aifs.py          read trimmed AIFS NetCDFs, → Zarr `aifs` var
scripts/05_compute_bias.py          all bias diagnostics, → outputs/bias/*.nc
scripts/06_make_plots.py            7 publication PNGs, → outputs/figures/
```

Each numbered step is idempotent / resumable. Failure of any one day is
logged and the loop continues — the resulting Zarr cube is partial but
correct over its populated subset.

### Per-day pipeline (concretely)

For a verifying IMD day `d`:

1. **IMERG** — download 48 half-hourly Final V07 granules covering the IMD day
   `(d 03:00 UTC, d+1 03:00 UTC)`. Sum to a daily total in mm/day on the IMERG
   native 0.1° grid. Conservatively regrid to the common 0.25° India grid.
2. **ERA5** — download hourly `tp` for the same window via CDS. Sum to a daily
   total. Conservatively regrid.
3. **AIFS** — init the forecast at `(d − 3 days, 03:00 UTC)` using ARCO-ERA5
   as IC source. Run 16 × 6-h steps. Select leads 78 / 84 / 90 / 96 h
   (corresponding to valid times `d 09 / 15 / 21 UTC` and `d+1 03 UTC`). Sum
   the four `tp06` values to get one IMD-day total in mm/day. Already on a
   regular 0.25° grid (Earth2Studio regrids AIFS's internal O96 reduced
   Gaussian), so the regrid step is mostly a coordinate alignment.
4. **Write** all three fields into the master Zarr store at time index `d`.

### Diagnostics computed

All on the populated subset of the 122-day cube:

- `mean_bias_map(lat, lon)` — `mean(forecast − observed)` over time.
- `rmse_map(lat, lon)` — `sqrt(mean((forecast − observed)^2))` over time.
- `bias_by_region(region)` — mean bias and RMSE within six rectangular
  meteorological regions (Western Ghats windward / leeward, Indo-Gangetic
  plain, Northeast, Himalayan foothills, peninsular interior).
- `bias_by_elevation(elev_bin)` — pointwise bias binned by ETOPO 2022 elevation
  (plain / foothill / low_mountain / high_mountain).
- `bias_by_rainfall_magnitude(rain_bin)` — pointwise bias binned by *observed*
  IMERG rainfall (trace / light / moderate / heavy / very_heavy, IMD
  thresholds).

Each runs for the AIFS−IMERG pair and again for the ERA5−IMERG baseline. The
AIFS−ERA5 residual is computed as a separate map.

Bias = `forecast − observed`. Positive = wet bias in the forecast.

### `bias_by_bsiso_phase` is the one diagnostic we did NOT compute

It was in the original design. The data source is broken:

- Kikuchi's IPRC real-time BSISO file (`rt_pc.txt`) was last updated
  2022-12-29. Useless for 2025.
- APCC's Lee et al. real-time product blocks direct URL access and would
  require an API arrangement.
- Self-computing the Kikuchi index from NOAA Interpolated OLR + IPRC's static
  EEOF vector files is tractable (~150 LoC, methodology in Kikuchi, Wang &
  Kajikawa 2012) but deferred — none of the three claims above depend on it.

The stub for the self-compute recipe is in `src/monsoon_bias/data/bsiso.py`
and the corresponding plot stub is in `analysis/plots.py`. Both raise
`NotImplementedError` and point at this section.

---

## 4. What we actually did (and what we didn't)

### Implemented and exercised

| Component | Notes |
|---|---|
| `data/era5.py` | CDS hourly `tp` downloader, idempotent file cache |
| `data/imerg.py` | EDL bearer-token + CMR search, manual redirect handling |
| `data/elevation.py` | NOAA ETOPO 2022 60-arc-sec via OPENDAP, conservatively regridded |
| `processing/accumulate.py` | IMD-day windowing for all three sources |
| `processing/regrid.py` | Conservative regrid for precip; refuses other methods |
| `processing/store.py` | Master Zarr, prealloc-NaN, single-chunk region writes |
| `forecast/run_aifs.py` | AIFS 96-h via Earth2Studio + ARCO-ERA5, lexicon patches |
| `forecast/trim.py` | Atomic in-place strip to `tp06` only (193× compression) |
| `analysis/bias.py` | Five non-deferred diagnostic functions |
| `analysis/plots.py` | Seven publication figures, cartopy + cmocean |

### Not implemented (and the reason)

- `data/bsiso.py` — deferred; live data sources are dormant or gated.
- `analysis.plot_bias_by_bsiso` — same.
- Land mask — not strictly needed for the three core findings, but the maps
  would be cleaner with one. Cheapest fix: `elevation >= 0` mask.
- Per-cell or per-region uncertainty quantification — nothing in the pipeline
  bootstraps or estimates confidence intervals on the bias values.

### Scale at writing

- **121 / 122** IMERG and ERA5 days populated (2025-09-30 missing because IMERG
  Final V07 had not fully published it at the time of download).
- **50 / 122** AIFS days populated. The remaining 72 require additional GPU
  sessions; the batch hit infrastructure trouble three times today
  (container-disk OOM from ARCO cache, GPU memory zombie, and a silent
  session kill on a third attempt). See `STATUS.md` lesson 10 and the resume
  checklist.

### Cost so far

~$19 GPU + storage across all sessions to date. Per the disk-full lesson and
the third silent kill, the remaining 72 forecasts would cost roughly $15 more
and take ~4 GPU hours on H100.

---

## 5. What we found

In one paragraph: AIFS over the 2025 Indian monsoon shows three independent,
structured, physically interpretable bias patterns — a coherent spatial map
with wet bias over Himalayan foothills and dry bias on the Western Ghats
windward face; a monotonic rainfall-magnitude miscalibration (over-predicts
trace / light, under-predicts heavy / very heavy); and a linear ~+0.8 mm/day
per km wet bias as a function of terrain elevation. The AIFS−ERA5 residual
map is itself structured, meaning AIFS does not merely inherit ERA5's
monsoon biases — it adds its own.

All three findings held when the sample was expanded from the first 21 days to
the current 50 days; magnitudes mostly sharpened rather than reverted.

**The three claims in the thesis are supported.** The precondition is met.
Full numbers are in `WRITEUP.md` § 3–5 and `notebooks/final_writeup.ipynb`.

---

## 6. What we cannot conclude (boundaries of the claim)

- **One model.** No comparison to GraphCast, Pangu, ECMWF operational HRES,
  or AI ensembles. The findings are about AIFS specifically; they may or
  may not generalize.
- **One season.** No multi-year validation. The 2025 monsoon was a relatively
  normal season; an anomalous year (strong El Niño, etc.) could exhibit
  different bias structure.
- **One region.** No comparison to West African or Maritime Continent
  monsoons. The orographic and tail-miscalibration patterns might be
  monsoon-universal or India-specific; we cannot tell.
- **No causal interpretation.** We can describe the bias structure but the
  diagnostics do not identify *why* AIFS produces it (loss function?
  architecture? training data smoothing? resolution?).
- **No uncertainty estimates.** Magnitudes are point estimates over the
  populated subset, not confidence intervals.

These are all addressable with additional work. None of them invalidate the
claim that the bias has learnable structure.

---

## 7. Next steps

### Immediate (≤ 1 week of laptop work, ~ 0 marginal GPU)

1. **Finish the AIFS batch** — ~$15 GPU, ~4 hours, gets to 122 / 122. The
   structural findings will not change; the magnitudes will tighten and the
   maps will smooth.
2. **Add a land mask** to the plots — replot 05 / 06 with ocean cells masked
   so the maps focus on what we care about.
3. **Add bootstrap CIs** to the bin-level findings — particularly the
   rainfall-magnitude tail bins, where the heavy / very-heavy bin counts are
   the smallest. Without CIs we can claim "structure exists" but not "the
   magnitudes are stable."

### Short-term (~ 2–4 weeks): the corrective head MVP

This is the immediate research-grade follow-on and the one that directly
exploits Proto 1's findings.

**Architecture.** Per-cell tabular regression (gradient-boosted trees,
likely LightGBM with quantile loss). Inputs: AIFS forecast value, a 3×3
spatial neighborhood of AIFS values, elevation, latitude / longitude,
day-of-year (sin/cos), and optionally an IMERG climatology baseline. Output:
corrected mm/day. Train on most of India; **hold out the Western Ghats** as
the spatial test set so the transfer story is sharp.

**Why this architecture before a CNN/UNet.** The smoking-gun finding (tail
miscalibration) is largely a per-cell calibration problem; most of the lift
should come from a per-cell function with a few spatial features. Tree
models also give you direct feature importance, which lets you check that
the model is actually exploiting the structure Proto 1 found (not random
features).

**Headline metric.** Not "% CRPS improvement"; *heavy-rain detection*. AIFS
currently misses a known fraction of >35 mm/day events. The corrected output
catches X% of those misses. That is the kind of result a non-specialist
operational user can act on.

**Reuse from Proto 1.** Master Zarr, elevation field, `config.REGIONS`,
`RAINFALL_BINS_MM`, all five `bias` functions for evaluation (compute the
same diagnostics on the corrected field; compare side-by-side with raw
AIFS), all seven plot functions.

**Estimated effort.** ~15–25 hours of focused dev. No GPU needed.
~80 LoC dataset construction + ~80 LoC train + ~50 LoC apply + a notebook.

### Medium-term (~ 1–3 months): the natural research questions

These are where the work starts to look like research rather than engineering:

1. **Does the corrective head transfer?**
   - Across years: train on 2025, evaluate on 2024.
   - Across regions: train on India, evaluate on West Africa / Maritime
     Continent / South America. This is the question with the highest
     research interest and the lowest risk of "I trained a model that
     memorized 2025 India."
2. **Does the loss function matter?** Vanilla MSE under-trains the tail.
   Quantile loss, distributional loss, focal MSE — comparison ablation.
   This is where there's a real methodological contribution.
3. **Multi-model corrective heads.** Train one corrective head that takes
   AIFS, GraphCast, and an ensemble member as inputs. Test whether the
   ensemble corrective head dominates any single-model correction.

### Long-term / PhD-scale

See the candid discussion in chat — separate question, addressed there.

---

## 8. Honest engineering / methodology weaknesses

For future reference, the things that are not great about how Proto 1 was
done that a thesis committee or reviewer would ask about:

- **Single-season validation.** All findings are 2025-specific. Cannot say
  anything about robustness across years.
- **No model comparison.** We claim "AI weather models have narrow
  distributions" via citation, but Proto 1 only validates that claim for one
  model.
- **No causal claim.** We describe the bias pattern but do not isolate
  whether it stems from the architecture, the loss, or the training data.
- **Subjective region boundaries.** `config.REGIONS` is six hand-drawn
  bounding boxes. A real analysis would use IMD subdivision shapefiles or
  cluster-derived regions.
- **Per-cell stratification can double-count.** When we compute bias by
  rainfall magnitude across (cell, day) pairs, neighboring cells on the same
  day are not independent. The effective sample size in each tail bin is
  smaller than the raw count suggests. Bootstrap CIs would expose this.
- **ERA5 as both training data and "baseline."** Comparing AIFS to ERA5 to
  isolate the AIFS-specific residual is correct in spirit but ERA5 is also
  what AIFS was trained on. The residual is, more precisely, "the part of
  AIFS's bias that is not in its training distribution as represented by
  ERA5 in this 2025 window." There is no perfect ground truth here.
- **The pipeline is in good shape but the code is prototype-grade.** No
  tests, no type checking enforced, no documentation site, no CI.

None of these are fatal. All are addressable in the follow-on work.

---

## 9. Where the artifacts live

- This document: `PROCESS.md`
- Brief: `WRITEUP.md`
- Reproducible notebook: `notebooks/final_writeup.ipynb`
- Status log + bootstrap lessons: `STATUS.md`
- All pipeline scripts: `scripts/`
- All analysis modules: `src/monsoon_bias/`
- Diagnostic NetCDFs: `outputs/bias/`
- Publication PNGs: `outputs/figures/`
- Master Zarr cube: `data/monsoon_bias.zarr` (not in git; ~13 MB local)
- AIFS NetCDFs: `data/forecasts/` (not in git; ~50 × 37 MB local)
- GitHub: <https://github.com/n0ndescript/forecasting-proto-1>
