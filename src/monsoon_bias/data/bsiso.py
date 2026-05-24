"""BSISO index — **DEFERRED** for this prototype.

Status (2026-05-21): the Kikuchi IPRC BSISO data files are stale (last
update 2022-10) and APCC's Lee et al. realtime product is not
publicly URL-accessible. We are skipping BSISO phase stratification in
the first pass of this prototype; the other stratifications (elevation,
region, rainfall magnitude) still run.

To pick this up later, the cleanest path is to **self-compute the
Kikuchi index** for 2025. The ingredients are still available:

* **EEOF vectors** (the projection basis for the index), JJASO season:
    https://iprc.soest.hawaii.edu/users/kazuyosh/ISO_index/data/olr.7917_01.25-90bpfil.JJASO.eeof_evec.nc
    (255 KB, last modified 2022-04-11 — fine, the basis is fixed)

* **NOAA Interpolated OLR** (the input field), daily, current:
    https://www.ncei.noaa.gov/data/outgoing-longwave-radiation-daily/

Algorithm (Kikuchi, Wang & Kajikawa 2012):
    1. Download daily Interpolated OLR for ~1979–2025.
    2. Subtract climatology (long-term daily mean).
    3. Bandpass-filter 25–90 days.
    4. Project filtered anomalies onto the saved EEOFs → PC1, PC2.
    5. Normalize by their season-specific stdev (also in the EOF file).
    6. amplitude = sqrt(PC1² + PC2²); phase = atan2(PC2, PC1) bucketed
       into 8 octants.
    7. Reassign phase = 0 when amplitude < 1.

Effort estimate: ~150 lines of NumPy/scipy/xarray; mostly the filter
spin-up handling. Not blocking the rest of the pipeline.

See also: alternative providers we ruled out for this pass —
    APCC (302 → error/method; needs API arrangement)
    NOAA MJO RMM (different index, not equivalent in summer)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Frozen findings from the URL probe on 2026-05-21; update if revisited.
KIKUCHI_RT_PC_URL = (
    "https://iprc.soest.hawaii.edu/users/kazuyosh/ISO_index/data/"
    "BSISO_25-90bpfil.rt_pc.txt"
)  # last updated 2022-12-29 — DO NOT use as-is for 2025 analysis.

KIKUCHI_EEOF_JJASO_URL = (
    "https://iprc.soest.hawaii.edu/users/kazuyosh/ISO_index/data/"
    "olr.7917_01.25-90bpfil.JJASO.eeof_evec.nc"
)


def download_bsiso(output_path: Path | None = None) -> Path:
    """**Deferred.** See module docstring."""
    raise NotImplementedError(
        "BSISO index is deferred for the first pass of this prototype. "
        "Kikuchi's pre-computed PC file is stale (ends 2022-12-29). To enable "
        "BSISO stratification, implement the self-compute path described in "
        "this module's docstring."
    )


def load_bsiso(path: Path | None = None) -> pd.DataFrame:
    """**Deferred.** See module docstring."""
    raise NotImplementedError("BSISO index is deferred. See module docstring.")
