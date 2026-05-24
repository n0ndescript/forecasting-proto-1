"""Lexicon patch for Earth2Studio 0.9.0 — adds CDS mappings for the 23
AIFS input variables that ship pre-lexicon.

Earth2Studio's later releases (0.14.0+) added these mappings, but
upgrading from 0.9.0 → 0.14.0 forces numpy 1.x → 2.x, zarr 2 → 3, and a
pandas downgrade — a level of dep churn that would risk breaking the
already-working IMERG/ERA5 ingest + Zarr store code. This module is a
contained alternative: import it and call :func:`apply_cds_lexicon_patch`
before instantiating ``earth2studio.data.CDS()`` and the missing
variables resolve correctly.

22 of 23 entries are copied verbatim from earth2studio 0.14.0's
lexicon. The 23rd (``zsl``) is AIFS's name for the surface
geopotential / orography; not present in v0.14.0's CDS lexicon either,
but it's the same CDS query as the existing ``z`` entry (single-level
geopotential).

Idempotent — calling twice does nothing harmful.
"""

from __future__ import annotations


# Source: NVIDIA/earth2studio v0.14.0 earth2studio/lexicon/cds.py
# (plus zsl mapped to surface geopotential, matching the existing `z` entry).
_MISSING_ENTRIES: dict[str, str] = {
    # Vertical velocity at every AIFS pressure level
    "w50":   "reanalysis-era5-pressure-levels::vertical_velocity::50",
    "w100":  "reanalysis-era5-pressure-levels::vertical_velocity::100",
    "w150":  "reanalysis-era5-pressure-levels::vertical_velocity::150",
    "w200":  "reanalysis-era5-pressure-levels::vertical_velocity::200",
    "w250":  "reanalysis-era5-pressure-levels::vertical_velocity::250",
    "w300":  "reanalysis-era5-pressure-levels::vertical_velocity::300",
    "w400":  "reanalysis-era5-pressure-levels::vertical_velocity::400",
    "w500":  "reanalysis-era5-pressure-levels::vertical_velocity::500",
    "w600":  "reanalysis-era5-pressure-levels::vertical_velocity::600",
    "w700":  "reanalysis-era5-pressure-levels::vertical_velocity::700",
    "w850":  "reanalysis-era5-pressure-levels::vertical_velocity::850",
    "w925":  "reanalysis-era5-pressure-levels::vertical_velocity::925",
    "w1000": "reanalysis-era5-pressure-levels::vertical_velocity::1000",
    # Surface fields
    "lsm":   "reanalysis-era5-single-levels::land_sea_mask::",
    "sdor":  "reanalysis-era5-single-levels::standard_deviation_of_orography::",
    "skt":   "reanalysis-era5-single-levels::skin_temperature::",
    "slor":  "reanalysis-era5-single-levels::slope_of_sub_gridscale_orography::",
    "tcw":   "reanalysis-era5-single-levels::total_column_water::",
    "zsl":   "reanalysis-era5-single-levels::geopotential::",     # AIFS name for orography
    # Soil temperature + moisture
    "stl1":  "reanalysis-era5-single-levels::soil_temperature_level_1::",
    "stl2":  "reanalysis-era5-single-levels::soil_temperature_level_2::",
    "swvl1": "reanalysis-era5-single-levels::volumetric_soil_water_layer_1::",
    "swvl2": "reanalysis-era5-single-levels::volumetric_soil_water_layer_2::",
}


def apply_cds_lexicon_patch() -> None:
    """Add the 23 missing CDS mappings AIFS needs.

    Idempotent. Safe to call multiple times — only adds keys that aren't
    already present so we don't clobber upstream fixes if anyone
    upgrades earth2studio later.
    """
    from earth2studio.lexicon.cds import CDSLexicon
    for k, v in _MISSING_ENTRIES.items():
        if k not in CDSLexicon.VOCAB:
            CDSLexicon.VOCAB[k] = v


# ARCO lexicon already has the 13 `w*` and `lsm` entries, so it's only
# missing 9 of AIFS's 94 inputs — all surface or soil. ARCO format is
# "zarr_variable_name::level" (empty level for single-level vars). Keys
# verified against the gcp-public-data-arco-era5 bucket's array list.
_MISSING_ARCO_ENTRIES: dict[str, str] = {
    "sdor":  "standard_deviation_of_orography::",
    "slor":  "slope_of_sub_gridscale_orography::",
    "skt":   "skin_temperature::",
    "tcw":   "total_column_water::",
    "zsl":   "geopotential_at_surface::",
    "stl1":  "soil_temperature_level_1::",
    "stl2":  "soil_temperature_level_2::",
    "swvl1": "volumetric_soil_water_layer_1::",
    "swvl2": "volumetric_soil_water_layer_2::",
}


def apply_arco_lexicon_patch() -> None:
    """Add the 9 missing ARCO mappings AIFS needs (surface + soil vars).

    ARCO is dramatically faster than CDS for batch IC fetching (direct
    GCS reads vs queued API requests), so use ARCO + this patch when
    running many forecasts. Idempotent.
    """
    from earth2studio.lexicon.arco import ARCOLexicon
    for k, v in _MISSING_ARCO_ENTRIES.items():
        if k not in ARCOLexicon.VOCAB:
            ARCOLexicon.VOCAB[k] = v
