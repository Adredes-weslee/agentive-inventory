from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd


def prefer_parquet(
    csv_path: str | Path,
    *,
    usecols: Optional[Iterable[str]] = None,
    dtypes: Optional[Mapping[str, str]] = None,
    memory_map: bool = True,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read a dataset by preferring a side-by-side Parquet file when available,
    falling back to the CSV otherwise. Callers can keep passing usecols/dtypes
    exactly as they do today.

    - If `<file>.parquet` exists next to `<file>.csv`, uses pd.read_parquet(..., columns=usecols)
    - Else, uses pd.read_csv(..., usecols=usecols, dtype=dtypes, memory_map=memory_map)

    NOTE: Parquet readers ignore dtypes; callers may still pass dtypes for CSV.
    """
    csvp = Path(csv_path)
    pqt = csvp.with_suffix(".parquet")

    if pqt.exists():
        df = pd.read_parquet(pqt, columns=usecols)
        if nrows is not None:
            return df.head(nrows)
        return df

    return pd.read_csv(
        csvp,
        usecols=usecols,
        dtype=dtypes,
        memory_map=memory_map,
        nrows=nrows,
    )
