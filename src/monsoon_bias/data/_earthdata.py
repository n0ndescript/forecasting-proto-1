"""Shared NASA Earthdata Login (EDL) helpers used by the IMERG downloader.

We avoid `earthaccess.login()` because URS's ``find_or_create_token``
endpoint is not provisioned for some new accounts. Using a manually
generated EDL bearer token + CMR + direct GES DISC downloads works
universally and survives the gap.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

from .. import config


CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"


class EarthdataError(RuntimeError):
    """Raised when EDL auth, CMR search, or GES DISC download fails."""


def load_edl_token() -> str:
    """Return the EDL bearer token from the ``EDL_TOKEN`` env var or
    ``.edl_token`` in the project root. Raises ``EarthdataError`` if not
    found.
    """
    token = os.environ.get("EDL_TOKEN")
    if not token:
        path = config.PROJECT_ROOT / ".edl_token"
        if path.exists():
            token = path.read_text().strip()
    if not token:
        raise EarthdataError(
            "No EDL bearer token. Set EDL_TOKEN env var or place a token at "
            f"{config.PROJECT_ROOT}/.edl_token (chmod 600). Generate one at "
            "https://urs.earthdata.nasa.gov/profile (Generate Token)."
        )
    return token.strip()


def cmr_search_granules(
    *,
    short_name: str,
    version: str,
    temporal: tuple[str, str],
    bounding_box: tuple[float, float, float, float],   # (W, S, E, N)
    page_size: int = 100,
    token: str | None = None,
) -> list[dict]:
    """Query CMR for granules matching the given collection + window.

    Returns the raw ``feed.entry`` list. Each entry's ``links`` field
    contains download URLs; use :func:`extract_data_urls` to pick out
    the data files.
    """
    if token is None:
        token = load_edl_token()
    w, s, e, n = bounding_box
    params = {
        "short_name": short_name,
        "version": version,
        "temporal": f"{temporal[0]},{temporal[1]}",
        "bounding_box": f"{w},{s},{e},{n}",
        "page_size": page_size,
        "sort_key": "start_date",
    }
    r = requests.get(
        CMR_GRANULES_URL,
        params=params,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    if r.status_code != 200:
        raise EarthdataError(f"CMR returned HTTP {r.status_code}: {r.text[:300]}")
    return r.json().get("feed", {}).get("entry", [])


def extract_data_urls(entries: list[dict], suffixes: tuple[str, ...] = (".HDF5", ".nc4")) -> list[str]:
    """Pick out the actual data-file URLs from CMR granule entries.

    GES DISC marks data links with ``rel`` containing ``data#`` and a
    filename ending in ``.HDF5`` (half-hourly) or ``.nc4`` (daily).
    """
    urls: list[str] = []
    for entry in entries:
        chosen: str | None = None
        for link in entry.get("links", []):
            href = link.get("href", "")
            rel = link.get("rel", "")
            if "data#" in rel and href.startswith("http") and href.endswith(suffixes):
                chosen = href
                break
        if chosen is None:
            for link in entry.get("links", []):
                href = link.get("href", "")
                if href.startswith("http") and href.endswith(suffixes):
                    chosen = href
                    break
        if chosen is not None:
            urls.append(chosen)
    return urls


def download_with_token(url: str, out: Path, *, token: str | None = None,
                        timeout: int = 600, max_retries: int = 3) -> Path:
    """Download a GES DISC file using an EDL bearer token.

    Re-attaches the Authorization header on each redirect because
    ``requests`` strips it on cross-host hops. Retries transient
    failures (5xx, connection errors) up to ``max_retries`` times with
    exponential backoff.
    """
    if token is None:
        token = load_edl_token()

    out.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(max_retries):
        try:
            current = url
            for _ in range(10):
                resp = session.get(current, headers=headers, stream=True,
                                   allow_redirects=False, timeout=timeout)
                if resp.status_code in (301, 302, 303, 307, 308):
                    current = resp.headers["Location"]
                    resp.close()
                    continue
                if resp.status_code >= 500:
                    raise EarthdataError(f"HTTP {resp.status_code} from {current}")
                resp.raise_for_status()
                tmp = out.with_suffix(out.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=256 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp.replace(out)
                return out
            raise EarthdataError(f"Too many redirects starting at {url}")
        except (requests.RequestException, EarthdataError) as exc:
            if attempt == max_retries - 1:
                raise EarthdataError(f"Download failed after {max_retries} attempts: {exc}") from exc
            time.sleep(2 ** attempt)
    raise EarthdataError("unreachable")
