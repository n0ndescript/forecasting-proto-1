# Does AIFS bias over the Indian monsoon have learnable structure?

**Status:** 50 of 122 forecast days analyzed (~41 % of season).
The headline structure is robust to additional days.
**Author:** Siddharth Vijayakrishnan · **Date:** 2026-05-28

---

## Headline

The precondition for a corrective-head follow-on is met. Across three independent
stratifications, AIFS's precipitation bias over India shows **structured,
geographically coherent, and physically interpretable patterns** — not white noise.

- **By rainfall magnitude:** AIFS over-predicts trace and light rain (+2.46 / +4.29
  mm/day) and under-predicts heavy and very-heavy events (−26.14 / −77.65 mm/day).
  This is the canonical "narrow predictive distribution" failure mode.
- **By elevation:** AIFS is dry over the Gangetic plain and consistently wet over
  every elevation bin above 500 m (+1.98 to +2.55 mm/day). Linear fit:
  **≈ +0.8 mm/day per km of elevation**.
- **By region:** Wet over the Himalayan foothills and the leeward Western Ghats;
  dry on the Western Ghats windward face. These are exactly the regions where a
  ~25 km global model's smoothed orography fails the hardest.

The AIFS − ERA5 residual (panel 3 of Figure 1) confirms these are not just
"reanalysis is hard" effects: AIFS departs from its own training distribution in
spatially structured ways.

---

## What we tested

**Question:** Does AIFS's bias over the Indian monsoon have structure that a
small corrective model could learn from? Or is the bias diffuse / unstructured /
predominantly random?

**Why it matters:** This is the precondition for building a "corrective head" —
a small ML model that learns to translate the AIFS forecast into a higher-skill
local product. If the bias has no structure, there's nothing to learn. If it has
clear structure, a corrective head is justified.

**Data and method:**

- **Target season:** 2025 Indian monsoon (Jun 1 – Sep 30, 122 days).
- **Forecast:** ECMWF AIFS-Single 1.1.0 via NVIDIA Earth2Studio. Initialized at
  03 UTC three days before each verifying date so the 6-h native step aligns
  cleanly with the IMD meteorological day (03 → 03 UTC). 96-h lead. ARCO-ERA5
  as the initial-condition source.
- **Ground truth:** NASA GPM IMERG Final V07 half-hourly, summed over the IMD day.
- **Baseline:** ERA5 hourly precipitation, accumulated the same way. Provides a
  three-way decomposition (AIFS − IMERG / ERA5 − IMERG / AIFS − ERA5) that
  isolates AIFS-specific error from shared reanalysis-vs-obs error.
- **Grid:** Conservatively regridded to a common 0.25° India grid (lat 6–38°N,
  lon 68–98°E). Conservative regridding is mandatory for precipitation —
  bilinear silently breaks mass conservation.

**Status snapshot at writing:** 50 / 122 AIFS forecasts complete (covering 2025
Jun 1 – Jul 26 plus mid-July gaps from the GPU batch's two infra interruptions);
121 / 122 IMERG and 121 / 122 ERA5 days populated in the master Zarr cube.
(The one missing IMERG day is 2025-09-30 — half-hourly granules not yet fully
published as of late May 2026.)

---

## Finding 1 — Spatial structure of the bias

![Three-panel bias decomposition](outputs/figures/03_three_panel_aifs_era5_residual.png)
*Figure 1. AIFS − IMERG (left), ERA5 − IMERG (middle), and the AIFS − ERA5
residual (right). 50-day sample. All three panels share a symmetric color
scale. Note the residual is not flat — AIFS departs from its training
distribution in coherent ways.*

The AIFS bias map shows three signatures simultaneously:

- **Wet bias along the Himalayan foothills** (red band running northwest to
  southeast across the top of the domain).
- **Dry bias on the Western Ghats windward face** (deep blue strip along the
  southwest coast). This is the classic orographic-precipitation miss for any
  global model coarser than ~10 km.
- **Coherent wet/dry alternation across the interior peninsula**, not random
  speckle.

The ERA5 bias map shows much of the same pattern with smaller amplitude — as
expected, since AIFS is trained on ERA5. **The right panel is the key one:** if
AIFS were just inheriting ERA5's biases, the residual would be near-zero noise.
It isn't. AIFS adds its own spatially structured error on top of the inherited
one, including a distinct wet bias over the Northeast.

---

## Finding 2 — Rainfall-magnitude tail miscalibration

![Bias by observed rainfall magnitude](outputs/figures/07_bias_by_rainfall_magnitude_aifs.png)
*Figure 2. Mean bias of AIFS forecasts conditional on the observed (IMERG) IMD-day
rainfall total. Each bar is averaged across all (cell, day) pairs in the bin;
counts shown. Bin edges follow IMD conventions: trace (0–1 mm), light (1–10),
moderate (10–35), heavy (35–75), very heavy (>75).*

This is the strongest single finding. AIFS over-predicts non-rain and light-rain
days, slightly under-predicts moderate events, and **catastrophically
under-predicts heavy and very-heavy events** (mean −27 and −75 mm/day in those
bins).

The sample is substantial in every bin including the tail (n = 36,063 for heavy,
n = 10,124 for very heavy). This isn't a few outlier cells — it's a systematic
narrowing of the precipitation distribution. The same pattern appears in
virtually every global AI weather model evaluated against radar / gauge truth in
the recent literature, but it's particularly stark over the monsoon.

A corrective head trained on this would directly target the wet / dry tails,
which is the highest-value calibration problem in monsoon nowcasting.

---

## Finding 3 — Orographic structure

The elevation stratification (Figure 3 — `05_bias_by_elevation_aifs.png`) shows
AIFS is dry over plains (−1.55 mm/day, n = 493k cell-days), wet at every higher
elevation bin (+1.98 to +2.55). The hexbin scatter
(`06_bias_vs_elevation_scatter_aifs.png`) yields a linear fit of
**≈ +0.8 mm/day per km of elevation**: AIFS systematically overshoots rainfall
the higher the terrain.

This is consistent with AIFS's underlying O96 reduced-Gaussian internal grid
(~25 km), which smooths out the sharp orographic gradients that drive real
monsoon precipitation peaks. The fix isn't a higher-resolution model — it's a
correction term keyed on the local elevation profile, which is exactly what a
corrective head can supply.

---

## Limitations

- **Sample size.** 50 / 122 forecast days at the time of writing (covering Jun 1
  – Jul 26 with a few mid-July gaps from two GPU-batch infra interruptions —
  see the resume checklist in `STATUS.md`). Moving from 21 → 50 days
  sharpened the wet-side magnitudes and left the dry tail bins almost
  unchanged, suggesting the headline structure is robust. The remaining
  72 forecasts can be added with a couple more GPU sessions.
- **No land mask yet.** Ocean cells are included in the figures. IMERG quality
  over ocean differs from land; the map figures should be re-rendered with a
  land mask before submission.
- **BSISO phase stratification deferred.** The plan was to also stratify by
  BSISO (active / break / transition phases of the monsoon). Kikuchi's
  real-time BSISO index file has been dormant since 2022; APCC's real-time
  product is gated. Self-computing the index from NOAA OLR + IPRC's static
  EEOF vectors is ~150 LoC and tractable but deferred. None of the three
  findings above depend on it.
- **2025 only.** Single-season evaluation. The structural findings are
  consistent with the multi-year literature on similar models, so we don't
  expect another year to invert them, but the magnitudes are 2025-specific.
- **AIFS only.** No GraphCast or Pangu comparison. GraphCast lacks native
  precipitation output (would require a separate diagnostic), and Pangu is on
  ECMWF's deprecation track. Adding a probabilistic ensemble model (AIFS-ENS,
  GenCast) is the natural Proto 2 extension.

---

## Recommendation for Proto 2

The three findings above answer the precondition question: **AIFS bias is
learnable, not noise.** This justifies the follow-on corrective-head project,
and it makes Proto 2 (regime-dependent ensemble skill) the right next test —
not because we need more bias evidence, but because the *operational* question
is which model to weight when, conditional on the atmospheric regime.

The BSISO data-source issue would force Proto 2 to either (a) self-compute the
BSISO index (~150 LoC, validation against pre-2022 published values), or
(b) substitute MJO RMM as the primary regime variable (NOAA / BoM both publish
real-time RMM, no API arrangement needed). RMM phases 2-3 and 5-6 align well
with monsoon active and break periods, so it's a defensible proxy and lowers
Proto 2's risk. Hunt's underlying question ("does model ranking depend on
regime?") is answerable with either.

**Estimated Proto 2 cost:** 50–100 hr H100, $150–300. ~500–800 LoC of new
analysis on top of the existing pipeline.

---

## Reproducibility

All code and pipeline scripts at <https://github.com/n0ndescript/forecasting-proto-1>.
The diagnostic NetCDFs behind every figure are in `outputs/bias/`; the raw
publication PNGs in `outputs/figures/`. Re-running on a populated master Zarr:

```bash
uv run python scripts/05_compute_bias.py
uv run python scripts/06_make_plots.py
```

Three-day-lead AIFS forecasts can be reproduced on any CUDA host with the
project's `[gpu]` extra installed (see `STATUS.md` for the specific torch /
flash-attn / earth2studio pins).
