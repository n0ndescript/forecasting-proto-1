"""GPM IMERG Final V07 downloader (half-hourly + daily).

Why half-hourly? The IMD "daily rainfall" window (03:00 UTC → 03:00 UTC)
does not align with the IMERG daily product's 00:00 UTC → 00:00 UTC
window. To get the correct 24-hour IMD-day total we sum the 48
half-hourly granules covering the IMD window.

Auth: EDL bearer token via :mod:`monsoon_bias.data._earthdata`.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

import pandas as pd

from .. import config
from . import _earthdata as edl

# Half-hourly Final Run V07
IMERG_HHR_SHORT_NAME = "GPM_3IMERGHH"
# Daily Final Run V07 (kept for sanity checks; not used in main pipeline)
IMERG_DAILY_SHORT_NAME = "GPM_3IMERGDF"
IMERG_VERSION = "07"

EXPECTED_GRANULES_PER_IMD_DAY = 48


def download_imerg_for_imd_day(
    verifying_date: pd.Timestamp,
    output_dir: Path | None = None,
    *,
    token: str | None = None,
    max_workers: int = 8,
) -> list[Path]:
    """Download all 48 IMERG half-hourly granules covering the IMD day
    ending on ``verifying_date``.

    The window runs from 03:00 UTC on ``verifying_date`` to 03:00 UTC
    the next day (inclusive of the start, exclusive of the end). Files
    that already exist locally are not re-downloaded.

    Downloads run in a thread pool of ``max_workers`` (default 8) since
    each granule is an independent HTTPS GET and the per-granule
    `download_with_token` creates its own ``requests.Session``. 8
    workers is well within GES DISC's tolerance and reduces wall-clock
    from ~3 min to ~30 s for the 48-file batch.

    Raises :class:`_earthdata.EarthdataError` if fewer than 48 files
    end up on disk (some part of the window is missing).
    """
    if token is None:
        token = edl.load_edl_token()
    if output_dir is None:
        output_dir = config.IMERG_DIR / verifying_date.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_utc, end_utc = config.imd_day_window_utc(verifying_date)
    # CMR temporal: subtract 1s from end so we don't accidentally pull
    # the next day's first granule (which begins exactly at end_utc).
    temporal = (
        start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        (end_utc - pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    lat_min, lat_max, lon_min, lon_max = config.INDIA_BBOX

    entries = edl.cmr_search_granules(
        short_name=IMERG_HHR_SHORT_NAME,
        version=IMERG_VERSION,
        temporal=temporal,
        bounding_box=(lon_min, lat_min, lon_max, lat_max),
        token=token,
    )
    urls = edl.extract_data_urls(entries, suffixes=(".HDF5",))
    if len(urls) < EXPECTED_GRANULES_PER_IMD_DAY:
        raise edl.EarthdataError(
            f"CMR returned {len(urls)} granules for "
            f"{verifying_date.date()}, expected {EXPECTED_GRANULES_PER_IMD_DAY}."
        )

    targets = [(url, output_dir / Path(url).name) for url in urls]
    pending = [(url, out) for url, out in targets if not out.exists()]

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(edl.download_with_token, url, out, token=token): (url, out)
                for url, out in pending
            }
            # Block on all futures; raise on the first failure so the caller
            # can decide whether to skip the day.
            for f in concurrent.futures.as_completed(futures):
                f.result()

    return sorted(out for _, out in targets)


def download_imerg_daily(verifying_date: pd.Timestamp,
                         output_dir: Path | None = None,
                         *, token: str | None = None) -> Path:
    """Download the IMERG daily Final V07 file for ``verifying_date``.

    Daily product window is 00:00 UTC → 00:00 UTC (not IMD-aligned), so
    this is only useful for quick sanity checks and the credentials test.
    """
    if token is None:
        token = edl.load_edl_token()
    if output_dir is None:
        output_dir = config.IMERG_DIR / "daily"
    output_dir.mkdir(parents=True, exist_ok=True)

    lat_min, lat_max, lon_min, lon_max = config.INDIA_BBOX
    entries = edl.cmr_search_granules(
        short_name=IMERG_DAILY_SHORT_NAME,
        version=IMERG_VERSION,
        temporal=(verifying_date.strftime("%Y-%m-%dT00:00:00Z"),
                  verifying_date.strftime("%Y-%m-%dT23:59:59Z")),
        bounding_box=(lon_min, lat_min, lon_max, lat_max),
        token=token,
    )
    urls = edl.extract_data_urls(entries, suffixes=(".nc4",))
    if not urls:
        raise edl.EarthdataError(f"No IMERG daily granule for {verifying_date.date()}.")
    out = output_dir / Path(urls[0]).name
    if not out.exists():
        edl.download_with_token(urls[0], out, token=token)
    return out
