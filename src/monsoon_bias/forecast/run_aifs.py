"""AIFS forecast pipeline via NVIDIA Earth2Studio 0.14.0.

GPU-bound. Intended to run on a CUDA host (RunPod H100 etc.) with the
``[gpu]`` extra installed:

    uv sync --extra gpu

Hard imports inside the entry function so this module is importable on a
CPU-only laptop (lets the analysis scripts read forecast output that
was produced elsewhere).

Key choices
-----------
* **Model:** ``AIFS.load_default_package()`` → AIFS-Single 1.1.0
  (released 2025-08-27, fixes the negative-precipitation issue present
  in AIFS 1.0.0). Verify the package version printed on first load.
* **Initial-condition source: ``ARCO``** (Google ARCO-ERA5 on GCS).
    - Fast: direct Zarr reads from GCS, no auth, no rate limit. Each
      IC fetch is ~5-30 s vs CDS's 10-30 min for the same data.
    - Earth2Studio 0.9.0's ``ARCOLexicon`` is missing 9 of AIFS's 94
      inputs (all surface/soil — ``sdor sslor skt tcw zsl stl1 stl2
      swvl1 swvl2``). :func:`_lexicon_patch.apply_arco_lexicon_patch`
      adds the 9 missing zarr-key mappings.
    - Earth2Studio's wrapper also has a defensive 2023-11-10 cutoff in
      ``_validate_time``; the underlying live bucket has data through
      at least 2026-04 (verified). :func:`_patch_arco_date_cutoff`
      lifts the wrapper check.
    - **CDS is the documented fallback** (also patched in
      ``_lexicon_patch``) if ARCO ever breaks — see
      :func:`apply_cds_lexicon_patch`.
* **Time step:** 6 h native (cannot be subdivided).
* **Init at 03 UTC.** Aligns 6-h step valid times (09/15/21/03 UTC) with
  the IMD day window 03→03 UTC after 4 steps.
* **Lead:** ``lead_days`` is the number of calendar days between init
  and the start of the verifying IMD day. We run for
  ``(lead_days + 1) * 4`` 6-h steps so the forecast extends through the
  end of the verifying day.
* **Output:** one NetCDF per init via :class:`NetCDF4Backend`.
* **Precip variable:** ``tp06`` (per-step 6-h accumulation, in meters).
  The accumulator in :mod:`monsoon_bias.processing.accumulate`
  defensively clips to zero in case Earth2Studio surfaces values from
  an older AIFS package.

Things to verify on the first real GPU run
------------------------------------------
(a) **Variable coverage.** AIFS needs ~89 input fields. Confirm ARCO
    provides all of them; Earth2Studio will fail loudly on the first run
    with the list of missing fields if anything is absent.
(b) **tp semantics.** Confirm ``tp06`` is per-step accumulation in m
    (not running total, not rate). If any negative values appear, that
    indicates the older AIFS 1.0.0 weights are being used and the
    accumulator's clip is needed.
(c) **Lead-time indexing.** ``run([t0], nsteps, ...)`` produces steps
    1..nsteps (step 0 = the IC itself, not a forecast). Confirm the
    output time coordinate values before slicing for the IMD day.
(d) **Spatial grid.** Earth2Studio outputs on a regular 0.25° lat/lon
    grid (AIFS's internal O96 reduced Gaussian is regridded). The
    accumulator + xarray-regrid pipeline assumes this. Confirm.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .. import config
from ..data.era5 import _load_cds_credentials


def _propagate_cds_creds_to_env() -> None:
    """Earth2Studio's CDS data source uses cdsapi, which reads
    CDSAPI_URL/CDSAPI_KEY env vars before falling back to ~/.cdsapirc.
    If a project-root .cdsapirc exists, expose it via env vars so we
    don't need to symlink into the home directory. Unused while we run
    on ARCO; kept for the CDS fallback path.
    """
    url, key, _ = _load_cds_credentials()
    if url and key:
        os.environ.setdefault("CDSAPI_URL", url)
        os.environ.setdefault("CDSAPI_KEY", key)


def _patch_arco_date_cutoff() -> None:
    """Remove Earth2Studio's 2023-11-10 cutoff on the ARCO data source.

    The live ``gcp-public-data-arco-era5`` bucket has continuous data
    through at least 2026-04 (verified directly on 2026-05-21). The
    cutoff is a conservative defensive check in the wrapper, not a
    constraint of the underlying dataset.

    Idempotent — safe to call multiple times.
    """
    from datetime import datetime as _dt
    from earth2studio.data import arco as _arco

    if getattr(_arco.ARCO, "_validate_time_patched", False):
        return

    def _validate_time_extended(self, times):
        for time in times:
            if not (time - _dt(1900, 1, 1)).total_seconds() % 3600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 1 hour interval for ARCO"
                )
            if time < _dt(year=1940, month=1, day=1):
                raise ValueError(
                    f"Requested date time {time} needs to be after January 1st, 1940 for ARCO"
                )
            # Original wrapper rejected time >= 2023-11-10; the live
            # bucket actually has data well past that, so we drop it.

    _arco.ARCO._validate_time = _validate_time_extended
    _arco.ARCO._validate_time_patched = True


def init_time_for_verifying_date(
    verifying_date: pd.Timestamp,
    lead_days: int = config.FORECAST_LEAD_DAYS,
) -> pd.Timestamp:
    """Return the AIFS init time for an IMD verifying date.

    Init is at 03:00 UTC, ``lead_days`` calendar days before the
    verifying date. For lead = 3, verifying = 2025-07-15 → init =
    2025-07-12 03:00 UTC.
    """
    return verifying_date.normalize() - pd.Timedelta(days=lead_days) + pd.Timedelta(hours=3)


def nsteps_for_lead(lead_days: int = config.FORECAST_LEAD_DAYS) -> int:
    """AIFS 6-h steps needed to cover lead + verifying day."""
    return (lead_days + 1) * 4   # = 16 for lead = 3 → 96-h forecast


def run_aifs_forecast(
    verifying_date: pd.Timestamp,
    output_path: Path | None = None,
    lead_days: int = config.FORECAST_LEAD_DAYS,
) -> Path:
    """Run a single deterministic AIFS forecast covering an IMD day.

    Returns the path to the output NetCDF. Skips the run if the file
    already exists.

    GPU is detected via ``torch.cuda``; raises if unavailable (CPU
    inference for AIFS is hours per step — never desirable).
    """
    init_time = init_time_for_verifying_date(verifying_date, lead_days=lead_days)
    nsteps = nsteps_for_lead(lead_days)

    if output_path is None:
        output_path = (config.FORECAST_DIR /
                       f"aifs_{init_time.strftime('%Y%m%dT%H%M')}_nsteps{nsteps}.nc")
    # A successful AIFS run produces ~50 MB of NetCDF. Anything tiny is
    # almost certainly an empty skeleton left behind when an earlier run
    # crashed after the backend was instantiated but before any data
    # was written — delete and re-run rather than silently using it.
    _MIN_VALID_NETCDF_BYTES = 1_000_000   # 1 MB; well above empty (~240 B)
    if output_path.exists():
        if output_path.stat().st_size >= _MIN_VALID_NETCDF_BYTES:
            return output_path
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Hard imports here so this module can be imported on CPU-only hosts
    # for the purpose of reading the forecast output.
    import torch
    from earth2studio.models.px import AIFS
    from earth2studio.data import ARCO
    from earth2studio.io import NetCDF4Backend
    from earth2studio.run import deterministic as run

    # earth2studio 0.14.0's ARCO lexicon covers most AIFS inputs but
    # still misses 5 surface/soil vars (tcw, swvl1, swvl2, stl1, stl2).
    # Our patch is idempotent and adds anything not already present.
    # Also lift the wrapper's 2023-11-11 _validate_time cutoff.
    from ._lexicon_patch import apply_arco_lexicon_patch
    apply_arco_lexicon_patch()
    _patch_arco_date_cutoff()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "AIFS requires CUDA; no GPU available. Re-run on the H100 host."
        )
    device = "cuda"

    package = AIFS.load_default_package()
    model = AIFS.load_model(package).to(device)
    data = ARCO()
    io = NetCDF4Backend(str(output_path))

    run([init_time.to_pydatetime()], nsteps, model, data, io)
    return output_path
