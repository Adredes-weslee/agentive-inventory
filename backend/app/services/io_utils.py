from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def prefer_parquet(
    csv_path: str | Path,
    parquet_path: Optional[str | Path] = None,
    *,
    columns: Optional[Iterable[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
    **csv_kwargs: Any,
) -> pd.DataFrame:
    """Load a dataset preferring Parquet with CSV fallback.

    Parameters
    ----------
    csv_path:
        Location of the canonical CSV file.
    parquet_path:
        Optional explicit Parquet path. When omitted we look for ``<csv>.parquet``.
    columns:
        Optional list/iterable of columns to read. Forwarded to the Parquet
        reader when available and mapped to ``usecols`` for CSV reads.
    dtype:
        Optional dtype mapping applied to the CSV fallback.
    csv_kwargs:
        Additional keyword arguments forwarded to :func:`pandas.read_csv`.
    """

    csv_path = Path(csv_path)
    pq_path = Path(parquet_path) if parquet_path is not None else csv_path.with_suffix(".parquet")

    column_list = list(columns) if columns is not None else None
    nrows = csv_kwargs.pop("nrows", None)

    if pq_path.exists():
        frame = pd.read_parquet(pq_path, columns=column_list)
        if nrows is not None:
            return frame.head(nrows)
        return frame

    csv_kwargs.setdefault("memory_map", True)
    if column_list is not None and "usecols" not in csv_kwargs:
        csv_kwargs["usecols"] = column_list

    if dtype is not None and "dtype" not in csv_kwargs:
        csv_kwargs["dtype"] = dtype

    if nrows is not None and "nrows" not in csv_kwargs:
        csv_kwargs["nrows"] = nrows

    return pd.read_csv(csv_path, **csv_kwargs)
