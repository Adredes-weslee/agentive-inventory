from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

DATA_DIR = Path(os.getenv("DATA_DIR", "data")).resolve()


def _prefer_parquet(csv_path: Path) -> Path:
    pqt = csv_path.with_suffix(".parquet")
    return pqt if pqt.exists() else csv_path


def load_table(
    filename: str,
    usecols: Optional[Sequence[str]] = None,
    dtype: Optional[dict] = None,
) -> pd.DataFrame:
    path = _prefer_parquet(DATA_DIR / filename)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=usecols)
    return pd.read_csv(path, usecols=usecols, dtype=dtype, memory_map=True)


def load_row_by_id(
    filename: str,
    id_value: str,
    id_col: str = "id",
    usecols: Optional[Sequence[str]] = None,
    dtype: Optional[dict] = None,
    chunksize: int = 50_000,
) -> Optional[pd.Series]:
    path = _prefer_parquet(DATA_DIR / filename)
    if path.suffix == ".parquet":
        import pyarrow.dataset as ds  # type: ignore[import-not-found]

        dataset = ds.dataset(path, format="parquet")
        table = dataset.to_table(columns=usecols, filter=ds.field(id_col) == id_value)
        df = table.to_pandas()
        return df.iloc[0] if not df.empty else None

    it = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunksize,
        memory_map=True,
    )
    for chunk in it:
        if id_col not in chunk.columns:
            continue
        hit = chunk[chunk[id_col] == id_value]
        if not hit.empty:
            return hit.iloc[0]
    return None


@lru_cache(maxsize=256)
def cached_row_by_id(filename: str, id_value: str) -> Optional[pd.Series]:
    return load_row_by_id(filename, id_value)
