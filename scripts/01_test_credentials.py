"""scripts/01_test_credentials.py

Smoke-test credential setup end-to-end:

  1. Locate CDS credentials (project root .cdsapirc or ~/.cdsapirc) and
     download one day of ERA5 2m temperature over India.
  2. Attempt NASA Earthdata login (via ~/.netrc). If it works, download
     one day of GPM IMERG Final V07 daily rainfall. If it doesn't,
     print clear registration instructions and continue.
  3. Plot whatever was successfully downloaded on a map of India with
     state boundaries.

Runs with no CLI args:  python scripts/01_test_credentials.py
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr

# Allow running as a plain script (no install) by adding src/ to sys.path.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from monsoon_bias import config  # noqa: E402

# Test date: mid-monsoon 2025. Far enough in the past for both ERA5 final
# (~3-month latency) and IMERG Final V07 (~3.5-month latency) to be
# fully published.
TEST_DATE = pd.Timestamp("2025-07-15")
TEST_OUT_DIR = config.DATA_DIR / "credentials_test"
TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = config.FIGURES_DIR / "01_credentials_test.png"


# ---------------------------------------------------------------------------
# CDS / ERA5
# ---------------------------------------------------------------------------

def _load_cds_credentials() -> tuple[str | None, str | None, Path | None]:
    """Look for .cdsapirc in the project root, then ~/.cdsapirc.
    Returns (url, key, source_path) or (None, None, None) if not found.
    """
    candidates = [config.PROJECT_ROOT / ".cdsapirc", Path.home() / ".cdsapirc"]
    for path in candidates:
        if not path.exists():
            continue
        url, key = None, None
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("url:"):
                url = line.split(":", 1)[1].strip()
            elif line.startswith("key:"):
                key = line.split(":", 1)[1].strip()
        if url and key:
            return url, key, path
    return None, None, None


def download_era5_t2m() -> Path | None:
    """Download one timestep of ERA5 2m temperature over India.
    Returns the NetCDF path on success, None on failure.
    """
    print("\n[ERA5] Locating CDS credentials...")
    url, key, src = _load_cds_credentials()
    if not (url and key):
        print("[ERA5] FAIL: no .cdsapirc found in project root or ~/.")
        print("       Get a key at https://cds.climate.copernicus.eu/ and place it in ~/.cdsapirc:")
        print("           url: https://cds.climate.copernicus.eu/api")
        print("           key: <uid>:<api-key>")
        return None
    print(f"[ERA5] Using credentials from {src}")

    try:
        import cdsapi
    except ImportError:
        print("[ERA5] FAIL: cdsapi not installed. Run `uv sync`.")
        return None

    out = TEST_OUT_DIR / f"era5_t2m_{TEST_DATE.date()}.nc"
    if out.exists():
        print(f"[ERA5] Already have {out.name}; skipping download.")
        return out

    lat_min, lat_max, lon_min, lon_max = config.INDIA_BBOX
    request = {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": [f"{TEST_DATE.year:04d}"],
        "month": [f"{TEST_DATE.month:02d}"],
        "day": [f"{TEST_DATE.day:02d}"],
        "time": ["12:00"],
        # CDS area order: [North, West, South, East]
        "area": [lat_max, lon_min, lat_min, lon_max],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    print(f"[ERA5] Requesting ERA5 t2m for {TEST_DATE.date()} 12:00 UTC over India...")
    try:
        client = cdsapi.Client(url=url, key=key)
        client.retrieve("reanalysis-era5-single-levels", request, str(out))
    except Exception as exc:  # noqa: BLE001 — surface any CDS error verbatim
        print(f"[ERA5] FAIL: {type(exc).__name__}: {exc}")
        return None

    print(f"[ERA5] OK — saved {out.name} ({out.stat().st_size / 1024:.1f} KB)")
    return out


# ---------------------------------------------------------------------------
# Earthdata / IMERG
# ---------------------------------------------------------------------------

EARTHDATA_HELP = """
[IMERG] Earthdata access not yet configured. To enable IMERG downloads:

  1. Register a free NASA Earthdata account:
         https://urs.earthdata.nasa.gov/users/new

  2. Accept the GES DISC application EULA from your Earthdata profile
     (Applications → Authorized Apps → "NASA GESDISC DATA ARCHIVE"):
         https://disc.gsfc.nasa.gov/earthdata-login

  3. Create ~/.netrc with your credentials and lock down permissions:
         echo "machine urs.earthdata.nasa.gov login <USER> password <PASS>" >> ~/.netrc
         chmod 600 ~/.netrc

  4. Re-run this script.
"""


def _load_edl_token() -> str | None:
    """Read an Earthdata Login bearer token from EDL_TOKEN env var or
    .edl_token in the project root. Returns None if not found.
    """
    env_token = os.environ.get("EDL_TOKEN")
    if env_token:
        return env_token.strip()
    path = config.PROJECT_ROOT / ".edl_token"
    if path.exists():
        return path.read_text().strip()
    return None


def _imerg_url_via_cmr(token: str) -> str | None:
    """Query CMR for the IMERG GPM_3IMERGDF.07 granule covering TEST_DATE
    and return its .nc4 download URL. Token is only needed for the
    download (CMR search is public), but we pass it for consistency.
    """
    lat_min, lat_max, lon_min, lon_max = config.INDIA_BBOX
    params = {
        "short_name": "GPM_3IMERGDF",
        "version": "07",
        "temporal": (f"{TEST_DATE.strftime('%Y-%m-%d')}T00:00:00Z,"
                     f"{TEST_DATE.strftime('%Y-%m-%d')}T23:59:59Z"),
        "bounding_box": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "page_size": 5,
    }
    r = requests.get(
        "https://cmr.earthdata.nasa.gov/search/granules.json",
        params=params, timeout=60,
        headers={"Authorization": f"Bearer {token}"},
    )
    r.raise_for_status()
    entries = r.json().get("feed", {}).get("entry", [])
    if not entries:
        return None
    for link in entries[0].get("links", []):
        href = link.get("href", "")
        rel = link.get("rel", "")
        # GES DISC data links have rel="...data#" and end in .nc4 or .HDF5
        if "data#" in rel and href.startswith("http") and href.endswith((".nc4", ".HDF5")):
            return href
    # Fallback: any http link with the right extension
    for link in entries[0].get("links", []):
        href = link.get("href", "")
        if href.startswith("http") and href.endswith((".nc4", ".HDF5")):
            return href
    return None


def _download_with_token(url: str, token: str, out: Path) -> Path:
    """Download a GES DISC file using an EDL bearer token. The Authorization
    header is re-applied on each redirect because requests strips it on
    cross-host hops.
    """
    session = requests.Session()
    headers = {"Authorization": f"Bearer {token}"}
    # Manually follow redirects so we can re-attach the token on each hop.
    current_url = url
    for _ in range(10):
        resp = session.get(current_url, headers=headers, stream=True, allow_redirects=False, timeout=600)
        if resp.status_code in (301, 302, 303, 307, 308):
            current_url = resp.headers["Location"]
            resp.close()
            continue
        resp.raise_for_status()
        with open(out, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
        return out
    raise RuntimeError(f"Too many redirects starting at {url}")


def download_imerg() -> Path | None:
    """Download one day of GPM IMERG Final V07 daily rainfall.

    Prefers a bearer token from EDL_TOKEN env or .edl_token (works even
    when the URS find_or_create_token endpoint is unavailable for a new
    account). Falls back to earthaccess + ~/.netrc if no token is set.
    """
    print("\n[IMERG] Locating Earthdata credentials...")
    token = _load_edl_token()

    if token:
        print("[IMERG] Using EDL bearer token (.edl_token / EDL_TOKEN).")
        try:
            url = _imerg_url_via_cmr(token)
        except Exception as exc:  # noqa: BLE001
            print(f"[IMERG] CMR search failed: {type(exc).__name__}: {exc}")
            return None
        if url is None:
            print(f"[IMERG] No granule URL found for {TEST_DATE.date()}.")
            return None
        out = TEST_OUT_DIR / Path(url).name
        if out.exists():
            print(f"[IMERG] Already have {out.name}; skipping download.")
            return out
        print(f"[IMERG] Downloading {Path(url).name} ...")
        try:
            _download_with_token(url, token, out)
        except Exception as exc:  # noqa: BLE001
            print(f"[IMERG] Download failed: {type(exc).__name__}: {exc}")
            return None
        print(f"[IMERG] OK — saved {out.name} ({out.stat().st_size / 1024:.1f} KB)")
        return out

    # No token — try earthaccess + netrc as a fallback.
    try:
        import earthaccess
    except ImportError:
        print("[IMERG] FAIL: earthaccess not installed. Run `uv sync`.")
        return None
    try:
        auth = earthaccess.login(strategy="netrc", persist=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[IMERG] Login error: {type(exc).__name__}: {exc}")
        print(EARTHDATA_HELP)
        return None
    if auth is None or not getattr(auth, "authenticated", False):
        print("[IMERG] Login did not authenticate.")
        print(EARTHDATA_HELP)
        return None
    print("[IMERG] Authenticated via netrc. Searching for granules...")
    lat_min, lat_max, lon_min, lon_max = config.INDIA_BBOX
    try:
        results = earthaccess.search_data(
            short_name="GPM_3IMERGDF", version="07",
            temporal=(TEST_DATE.strftime("%Y-%m-%d"), TEST_DATE.strftime("%Y-%m-%d")),
            bounding_box=(lon_min, lat_min, lon_max, lat_max),
        )
        files = earthaccess.download(results, str(TEST_OUT_DIR)) if results else []
    except Exception as exc:  # noqa: BLE001
        print(f"[IMERG] Search/download failed: {type(exc).__name__}: {exc}")
        return None
    if not files:
        print("[IMERG] No files downloaded.")
        return None
    path = Path(files[0])
    print(f"[IMERG] OK — saved {path.name} ({path.stat().st_size / 1024:.1f} KB)")
    return path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _setup_india_axis(ax) -> None:
    """Add coastlines, borders, and (best-effort) state boundaries."""
    import cartopy.crs as ccrs  # noqa: F401 — imported for side effect in tests
    import cartopy.feature as cfeature

    lat_min, lat_max, lon_min, lon_max = config.INDIA_BBOX
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="--")
    try:
        states = cfeature.NaturalEarthFeature(
            "cultural", "admin_1_states_provinces_lines", "10m",
            facecolor="none", edgecolor="gray", linewidth=0.3,
        )
        ax.add_feature(states)
    except Exception:  # noqa: BLE001 — Natural Earth download can fail offline
        pass
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False


def _open_era5(path: Path) -> xr.DataArray:
    ds = xr.open_dataset(path)
    # CDS Beta varies in variable name: t2m, 2t, or 2m_temperature.
    name = next((n for n in ("t2m", "2t", "2m_temperature") if n in ds), None)
    if name is None:
        raise RuntimeError(f"No t2m-like variable in {path.name}: {list(ds.data_vars)}")
    da = ds[name].squeeze() - 273.15  # K → °C
    da.attrs["units"] = "°C"
    # Standardize coord names if present.
    rename = {}
    if "latitude" in da.coords:
        rename["latitude"] = "lat"
    if "longitude" in da.coords:
        rename["longitude"] = "lon"
    return da.rename(rename) if rename else da


def _open_imerg(path: Path) -> xr.DataArray:
    ds = xr.open_dataset(path)
    if "precipitation" not in ds:
        raise RuntimeError(f"No 'precipitation' in {path.name}: {list(ds.data_vars)}")
    da = ds["precipitation"].squeeze()
    # IMERG daily V07 stores the daily *mean rate* in mm/hr; convert to mm/day.
    if str(da.attrs.get("units", "")).lower() in ("mm/hr", "mm h-1", "mm hr-1"):
        da = da * 24.0
        da.attrs["units"] = "mm/day"
    # Native dims are (lon, lat); transpose to (lat, lon).
    if "lat" in da.dims and "lon" in da.dims:
        da = da.transpose("lat", "lon")
    elif "latitude" in da.dims and "longitude" in da.dims:
        da = da.rename({"latitude": "lat", "longitude": "lon"}).transpose("lat", "lon")
    return da


@dataclass
class PlotInputs:
    era5: xr.DataArray | None
    imerg: xr.DataArray | None


def make_plot(inputs: PlotInputs) -> Path | None:
    import cartopy.crs as ccrs

    panels = [(name, da) for name, da in (("era5", inputs.era5), ("imerg", inputs.imerg)) if da is not None]
    if not panels:
        print("\n[PLOT] Nothing to plot — both downloads failed.")
        return None

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1, len(panels), figsize=(6.5 * len(panels), 6.5),
        subplot_kw={"projection": proj}, constrained_layout=True,
    )
    if len(panels) == 1:
        axes = [axes]

    for ax, (name, da) in zip(axes, panels):
        _setup_india_axis(ax)
        if name == "era5":
            mesh = ax.pcolormesh(
                da["lon"], da["lat"], da.values,
                transform=proj, cmap="RdYlBu_r", shading="auto",
            )
            cb = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.06, shrink=0.85)
            cb.set_label("2m temperature (°C)")
            ax.set_title(f"ERA5 t2m — {TEST_DATE.date()} 12:00 UTC")
        else:
            # Mask trace values to make the structure visible.
            data = np.where(da.values < 0.1, np.nan, da.values)
            mesh = ax.pcolormesh(
                da["lon"], da["lat"], data,
                transform=proj, cmap="Blues", shading="auto",
                vmin=0, vmax=max(50.0, float(np.nanpercentile(data, 99))),
            )
            cb = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.06, shrink=0.85)
            cb.set_label("Daily rainfall (mm/day)")
            ax.set_title(f"IMERG Final V07 — {TEST_DATE.date()}")

    fig.suptitle("Credentials smoke test — India domain", fontsize=12)
    fig.savefig(FIG_PATH, dpi=200)
    plt.close(fig)
    print(f"\n[PLOT] Saved {FIG_PATH}")
    return FIG_PATH


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print(f"Credentials smoke test  |  test date: {TEST_DATE.date()}")
    print(f"Project root: {config.PROJECT_ROOT}")
    print("=" * 72)

    era5_path = download_era5_t2m()
    imerg_path = download_imerg()

    era5_da, imerg_da = None, None
    if era5_path is not None:
        try:
            era5_da = _open_era5(era5_path)
        except Exception:  # noqa: BLE001
            print("[ERA5] Could not open downloaded NetCDF:")
            traceback.print_exc()
    if imerg_path is not None:
        try:
            imerg_da = _open_imerg(imerg_path)
        except Exception:  # noqa: BLE001
            print("[IMERG] Could not open downloaded NetCDF:")
            traceback.print_exc()

    make_plot(PlotInputs(era5=era5_da, imerg=imerg_da))

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  ERA5 (CDS):       {'OK' if era5_da is not None else 'FAILED'}")
    print(f"  IMERG (Earthdata):{'OK' if imerg_da is not None else 'FAILED'}")
    if imerg_da is None:
        print("\n  IMERG missing is expected if you haven't registered Earthdata yet.")
        print("  See the [IMERG] instructions above to set it up.")

    # Exit non-zero only if BOTH failed — partial success is the expected
    # state on first run.
    return 0 if (era5_da is not None or imerg_da is not None) else 1


if __name__ == "__main__":
    sys.exit(main())
