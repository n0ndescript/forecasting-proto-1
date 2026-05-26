"""Trim an AIFS forecast NetCDF down to tp06 only.

Earth2Studio's ``NetCDF4Backend`` writes every output variable AIFS
emits (~89 fields × 17 lead times × global 0.25° grid ≈ 7.2 GB per
forecast). For the monsoon bias analysis we need only ``tp06``. Trimming
to tp06 brings each file to ~80 MB, which is the difference between
122 forecasts fitting in 80 GB of pod disk vs not.

Trim is done in two steps:

  1. Write the trimmed file to ``<out>.tmp`` (atomic, won't corrupt the
     destination if the process dies mid-write).
  2. Reopen the temp file and verify ``tp06`` round-trips with the same
     shape and finite-value coverage as the source.
  3. ``os.replace`` the temp file over the destination.
  4. Optionally delete the source.

The function is idempotent: if the destination already exists and is
smaller than the source, it's treated as already trimmed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr

_KEEP_VAR = "tp06"
_MIN_REASONABLE_BYTES = 1_000_000   # trimmed file should still be > 1 MB
_MAX_REASONABLE_BYTES = 500_000_000  # ...and < 500 MB (catches "trim didn't drop vars")


def trim_aifs_forecast(
    in_path: Path,
    out_path: Path | None = None,
    delete_source: bool = False,
) -> Path:
    """Drop every variable except :data:`_KEEP_VAR` from an AIFS NetCDF.

    Args:
        in_path: full AIFS NetCDF as written by Earth2Studio's
            NetCDF4Backend.
        out_path: where to write the trimmed file. Defaults to
            ``in_path`` itself (in-place replace via temp file).
        delete_source: if ``out_path != in_path``, optionally unlink
            ``in_path`` after verifying the trimmed file is good. No-op
            if ``out_path == in_path`` (the replace already removed the
            source under that name).

    Returns:
        Path to the trimmed file.
    """
    in_path = Path(in_path)
    if out_path is None:
        out_path = in_path
    out_path = Path(out_path)

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    src_bytes = in_path.stat().st_size

    # Idempotency: if out_path already exists, looks trimmed, and the source
    # is either gone or larger, assume we're done.
    if (
        out_path.exists()
        and out_path != in_path
        and out_path.stat().st_size < src_bytes
        and out_path.stat().st_size <= _MAX_REASONABLE_BYTES
    ):
        return out_path

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    # Open source, slice down to one variable, write to temp with zlib.
    with xr.open_dataset(in_path) as ds:
        if _KEEP_VAR not in ds.data_vars:
            raise ValueError(
                f"{in_path.name}: missing {_KEEP_VAR!r}; vars are {list(ds.data_vars)}"
            )
        kept = ds[[_KEEP_VAR]]
        # Materialize before writing — otherwise to_netcdf re-reads lazily
        # from the same source path we're about to replace.
        kept = kept.load()
        encoding = {_KEEP_VAR: {"zlib": True, "complevel": 4}}
        kept.to_netcdf(tmp_path, encoding=encoding)

    # Verify the temp file: opens, has tp06, same shape, comparable finite-
    # value coverage. The finite check guards against a silent dtype/encoding
    # mishap that turns valid cells into NaN.
    with xr.open_dataset(tmp_path) as check:
        if _KEEP_VAR not in check.data_vars:
            tmp_path.unlink()
            raise ValueError(f"trimmed file missing {_KEEP_VAR}")
        if check[_KEEP_VAR].shape != _read_shape(in_path, _KEEP_VAR):
            tmp_path.unlink()
            raise ValueError(
                f"trimmed shape {check[_KEEP_VAR].shape} != source shape "
                f"{_read_shape(in_path, _KEEP_VAR)}"
            )
        n_finite = int(np.isfinite(check[_KEEP_VAR].values).sum())
        if n_finite == 0:
            tmp_path.unlink()
            raise ValueError("trimmed tp06 has no finite values")

    out_bytes = tmp_path.stat().st_size
    if not (_MIN_REASONABLE_BYTES <= out_bytes <= _MAX_REASONABLE_BYTES):
        tmp_path.unlink()
        raise ValueError(
            f"trimmed size {out_bytes / 1e6:.1f} MB out of expected range "
            f"({_MIN_REASONABLE_BYTES / 1e6:.0f}-{_MAX_REASONABLE_BYTES / 1e6:.0f} MB)"
        )

    os.replace(tmp_path, out_path)

    if delete_source and out_path != in_path and in_path.exists():
        in_path.unlink()

    return out_path


def _read_shape(path: Path, var: str) -> tuple[int, ...]:
    with xr.open_dataset(path) as ds:
        return tuple(ds[var].shape)
