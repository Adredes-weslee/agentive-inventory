r"""backend\app\services\inventory_service.py

Inventory helper utilities built on top of the M5 dataset."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

import pandas as pd

from .io_utils import load_row_by_id, load_table

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SKUPrice:
    """Lightweight container for SKU price metadata."""

    item_id: str
    store_id: Optional[str]
    sell_price: float
    week: Optional[int]


class InventoryService:
    """Provide SKU metadata, unit cost estimates and inventory placeholders."""

    def __init__(self, data_root: str = "data", config_root: Optional[str] = None) -> None:
        env_root = os.getenv("DATA_DIR")
        self.data_root = data_root if data_root else (env_root or "data")
        self.config_root = config_root or "configs"
        self.settings: Dict[str, Any] = {}
        self._sales_columns: Optional[list[str]] = None
        self._load_settings()

    # ------------------------------------------------------------------
    def _load_settings(self) -> None:
        default_settings: Dict[str, Any] = {
            "lead_time_days": 7,
            "gross_margin_rate": 0.0,
            "lead_time_overrides": {},
        }

        settings_path = os.path.join(self.config_root, "settings.yaml")
        self.settings = default_settings.copy()

        if not os.path.exists(settings_path):
            LOGGER.debug("Settings file not found at %s; using defaults.", settings_path)
            return

        if yaml is None:
            LOGGER.warning("PyYAML is not installed; unable to read %s. Using defaults.", settings_path)
            return

        try:
            with open(settings_path, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
        except Exception as exc:  # pragma: no cover - defensive logging only
            LOGGER.warning("Unable to load settings from %s: %s", settings_path, exc)
            return

        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if value is not None:
                    self.settings[key] = value
        else:  # pragma: no cover - defensive logging only
            LOGGER.warning("Settings at %s are not a mapping; using defaults.", settings_path)

    # ------------------------------------------------------------------
    def _data_path(self, filename: str) -> Path:
        path = Path(self.data_root) / filename
        return path

    # ------------------------------------------------------------------
    def _sales_dtype(self, columns: Optional[Sequence[str]] = None) -> Dict[str, str]:
        dtype: Dict[str, str] = {}
        base_string = {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id"}
        if columns is None:
            for col in base_string:
                dtype[col] = "string"
            return dtype

        for col in columns:
            if col in base_string:
                dtype[col] = "string"
            if col.startswith("d_"):
                dtype[col] = "float32"
        for col in base_string:
            if col not in dtype:
                dtype[col] = "string"
        return dtype

    # ------------------------------------------------------------------
    def _price_dtype(self, columns: Optional[Sequence[str]] = None) -> Dict[str, str]:
        dtype: Dict[str, str] = {}
        string_cols = {"item_id", "store_id"}
        numeric_cols = {"sell_price": "float32", "wm_yr_wk": "int32"}

        if columns is None:
            for col in string_cols:
                dtype[col] = "string"
            dtype.update(numeric_cols)
            return dtype

        for col in columns:
            if col in string_cols:
                dtype[col] = "string"
            if col in numeric_cols:
                dtype[col] = numeric_cols[col]
        return dtype

    # ------------------------------------------------------------------
    def _sales_columns_list(self) -> list[str]:
        if self._sales_columns is None:
            sales_path = self._data_path("sales_train_validation.csv")
            if not sales_path.exists():
                self._sales_columns = []
            else:
                header = pd.read_csv(sales_path, nrows=0)
                self._sales_columns = list(header.columns)
        return self._sales_columns

    # ------------------------------------------------------------------
    def _load_sales(self, usecols: list[str]) -> pd.DataFrame:
        columns = list(dict.fromkeys(usecols))
        dtype = self._sales_dtype(columns)
        return load_table(str(self._data_path("sales_train_validation.csv")), usecols=columns, dtype=dtype)

    # ------------------------------------------------------------------
    def _load_calendar(self, usecols: Optional[list[str]] = None) -> pd.DataFrame:
        columns = list(dict.fromkeys(usecols or [])) if usecols else None
        dtype = {"d": "string"} if columns and "d" in columns else None
        return load_table(str(self._data_path("calendar.csv")), usecols=columns, dtype=dtype)

    # ------------------------------------------------------------------
    def _load_prices(self, usecols: Optional[list[str]] = None) -> pd.DataFrame:
        columns = list(dict.fromkeys(usecols or [])) if usecols else None
        dtype = self._price_dtype(columns)
        return load_table(str(self._data_path("sell_prices.csv")), usecols=columns, dtype=dtype)

    # ------------------------------------------------------------------
    def _load_price_rows(
        self,
        item_id: str,
        store_id: Optional[str] = None,
        usecols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        base_cols = ["item_id", "sell_price", "wm_yr_wk"]
        if store_id:
            base_cols.append("store_id")
        if usecols:
            base_cols.extend(usecols)
        columns = list(dict.fromkeys(base_cols))
        path = self._data_path("sell_prices.csv")
        dtype = self._price_dtype(columns)

        parquet_path = path.with_suffix(".parquet")
        if parquet_path.exists():
            df = load_table(str(path), usecols=columns, dtype=dtype)
            df = df[df["item_id"] == item_id]
            if store_id and "store_id" in df.columns:
                df = df[df["store_id"] == store_id]
            return df

        if not path.exists():
            return pd.DataFrame(columns=columns)

        matches = []
        for chunk in pd.read_csv(
            path,
            usecols=columns,
            dtype=dtype,
            chunksize=50_000,
            memory_map=True,
        ):
            filtered = chunk[chunk["item_id"] == item_id]
            if store_id and "store_id" in filtered.columns:
                filtered = filtered[filtered["store_id"] == store_id]
            if not filtered.empty:
                matches.append(filtered)

        if matches:
            return pd.concat(matches, ignore_index=True)
        return pd.DataFrame(columns=columns)

    # ------------------------------------------------------------------
    def get_row(self, sku_id: str, usecols: list[str]) -> Optional[pd.Series]:
        columns = list(dict.fromkeys(["id", *usecols]))
        dtype = self._sales_dtype(columns)
        row = load_row_by_id(
            str(self._data_path("sales_train_validation.csv")),
            sku_id,
            id_col="id",
            usecols=columns,
            dtype=dtype,
        )
        if row is not None:
            return row

        alt_columns = list(dict.fromkeys(["item_id", *usecols]))
        alt_dtype = self._sales_dtype(alt_columns)
        return load_row_by_id(
            str(self._data_path("sales_train_validation.csv")),
            sku_id,
            id_col="item_id",
            usecols=alt_columns,
            dtype=alt_dtype,
        )

    # ------------------------------------------------------------------
    def list_ids(self, limit: int) -> list[str]:
        limit = max(0, int(limit))
        if limit == 0:
            return []

        path = self._data_path("sales_train_validation.csv")
        ids: list[str] = []
        parquet_path = path.with_suffix(".parquet")

        if parquet_path.exists():
            df = self._load_sales(["id"])
            ids = [str(x) for x in df["id"].dropna().astype(str).tolist() if str(x).strip()]
            return ids[:limit]

        if not path.exists():
            return []

        reader = pd.read_csv(
            path,
            usecols=["id"],
            dtype={"id": "string"},
            chunksize=limit,
            memory_map=True,
        )

        if isinstance(reader, pd.DataFrame):
            chunk_ids = [str(x) for x in reader["id"].dropna().tolist() if str(x).strip()]
            return chunk_ids[:limit]

        for chunk in reader:
            chunk_ids = [str(x) for x in chunk["id"].dropna().tolist() if str(x).strip()]
            ids.extend(chunk_ids)
            if len(ids) >= limit:
                break

        return ids[:limit]

    # ------------------------------------------------------------------
    def has_sku(self, sku_id: str) -> bool:
        """Return ``True`` when the SKU exists in the sales dataset."""

        try:
            return self.get_row(sku_id, ["id"]) is not None
        except Exception:  # pragma: no cover - defensive logging only
            LOGGER.warning("Unable to evaluate SKU existence for %s", sku_id)
            return False

    # ------------------------------------------------------------------
    def sku_exists(self, sku_id: str) -> bool:
        return self.has_sku(sku_id)

    # ------------------------------------------------------------------
    def get_sku_info(self, sku_id: str) -> Dict[str, Any]:
        row = self.get_row(
            sku_id,
            ["item_id", "dept_id", "cat_id", "store_id", "state_id"],
        )
        if row is None:
            return {}
        rec = row.to_dict()
        return {
            "item_id": rec.get("item_id"),
            "dept_id": rec.get("dept_id"),
            "cat_id": rec.get("cat_id"),
            "store_id": rec.get("store_id"),
            "state_id": rec.get("state_id"),
        }

    # ------------------------------------------------------------------
    def get_current_inventory(self, sku_id: str) -> int:
        """Return stub on-hand inventory levels."""

        return 0

    # ------------------------------------------------------------------
    @lru_cache(maxsize=1024)
    def get_latest_price(self, sku_id: str) -> Optional[SKUPrice]:
        """Return the most recent sell price for the SKU."""

        price_rows = self._load_price_rows(sku_id)
        if price_rows.empty:
            return None

        latest = price_rows.sort_values("wm_yr_wk").iloc[-1]
        return SKUPrice(
            item_id=sku_id,
            store_id=latest.get("store_id"),
            sell_price=float(latest.get("sell_price", 0.0)),
            week=int(latest.get("wm_yr_wk")) if not pd.isna(latest.get("wm_yr_wk")) else None,
        )

    # ------------------------------------------------------------------
    def _get_price_series(self, sku_id: str, store_id: Optional[str] = None) -> Optional[pd.Series]:
        price_rows = self._load_price_rows(sku_id, store_id=store_id)
        if price_rows.empty:
            return pd.Series(dtype=float)

        return price_rows["sell_price"].astype(float)

    # ------------------------------------------------------------------
    def get_unit_cost(self, sku_id: str) -> float:
        """Return a proxy unit cost using median sell price data."""

        fallback_cost = 10.0
        store_id: Optional[str] = None
        lookup_item_id: Optional[str] = None

        # Prefer metadata from the sales table when available
        sku_info = self.get_sku_info(sku_id)
        if sku_info:
            lookup_item_id = sku_info.get("item_id") or None
            store_id = sku_info.get("store_id")

        # Fallback: parse composite identifiers from the SKU id itself
        if lookup_item_id is None:
            parsed_item, parsed_store = self._parse_ids(sku_id)
            lookup_item_id = parsed_item
            store_id = store_id or parsed_store

        # Price data is keyed by item identifier, not the composite row id
        price_series = self._get_price_series(lookup_item_id or sku_id, store_id)
        if price_series is None or price_series.empty:
            LOGGER.warning(
                "Falling back to default unit cost for SKU %s; median price unavailable.",
                sku_id,
            )
            return fallback_cost

        median_price = float(price_series.median())
        if median_price <= 0:
            LOGGER.warning(
                "Median sell price for SKU %s was non-positive; using fallback %.2f.",
                sku_id,
                fallback_cost,
            )
            return fallback_cost

        return median_price

    # ------------------------------------------------------------------
    def get_lead_time_days(self, sku_id: str) -> int:
        """Return supplier lead time with optional category overrides."""

        default_lead_time = int(self.settings.get("lead_time_days", 7))
        overrides = self.settings.get("lead_time_overrides", {})
        lead_time = default_lead_time

        if isinstance(overrides, dict) and overrides:
            sku_info = self.get_sku_info(sku_id)
            if sku_info:
                for key in ("item_id", "dept_id", "cat_id", "store_id"):
                    lookup_key = sku_id if key == "item_id" else sku_info.get(key)
                    if lookup_key is None:
                        continue
                    override = overrides.get(lookup_key)
                    if override is not None:
                        lead_time = int(override)
                        break

        if lead_time <= 0:
            LOGGER.warning(
                "Configured lead time for SKU %s was non-positive; using default %s",
                sku_id,
                default_lead_time,
            )
            lead_time = default_lead_time

        return lead_time

    # ------------------------------------------------------------------
    def get_price(self, sku_id: str) -> float:
        """Estimate a sell price using historical mean and configured margin uplift."""

        margin_rate = float(self.settings.get("gross_margin_rate", 0.0))
        store_id: Optional[str] = None
        lookup_item_id: Optional[str] = None

        sku_info = self.get_sku_info(sku_id)
        if sku_info:
            lookup_item_id = sku_info.get("item_id") or None
            store_id = sku_info.get("store_id")

        if lookup_item_id is None:
            parsed_item, parsed_store = self._parse_ids(sku_id)
            lookup_item_id = parsed_item
            store_id = store_id or parsed_store

        price_series = self._get_price_series(lookup_item_id or sku_id, store_id)
        if price_series is None or price_series.empty:
            LOGGER.warning(
                "Sell price data unavailable for SKU %s; deriving price from unit cost.",
                sku_id,
            )
            base_price = self.get_unit_cost(sku_id)
        else:
            base_price = float(price_series.mean())

        if base_price <= 0:
            base_price = 10.0
            LOGGER.warning(
                "Base price for SKU %s was non-positive; using fallback %.2f.",
                sku_id,
                base_price,
            )

        return base_price * (1 + margin_rate)

    # ------------------------------------------------------------------
    def estimate_unit_cost(self, sku_id: str) -> float:
        """Backward compatible wrapper for legacy callers."""

        return self.get_unit_cost(sku_id)

    # ------------------------------------------------------------------
    def _parse_ids(self, sku_id: str) -> tuple[Optional[str], Optional[str]]:
        """Parse item and store identifiers from composite SKU identifiers."""

        if not sku_id:
            return None, None
        parts = str(sku_id).split("_")
        item_id = "_".join(parts[:3]) if len(parts) >= 3 else None
        store_id = "_".join(parts[3:5]) if len(parts) >= 5 else None
        item_id = item_id or None
        store_id = store_id or None
        return item_id, store_id

    # ------------------------------------------------------------------
    def get_unit_price(self, sku_id: str) -> Optional[float]:
        """Return the median historical sell price for the SKU's item."""

        item_id, store_id = self._parse_ids(sku_id)

        if item_id is None:
            sku_info = self.get_sku_info(sku_id)
            item_id = sku_info.get("item_id") if sku_info else sku_id

        if item_id is None:
            return None

        price_rows = self._load_price_rows(item_id, store_id=store_id)
        if price_rows.empty or "sell_price" not in price_rows:
            return None

        median_price = float(price_rows["sell_price"].median())

        if not math.isfinite(median_price) or median_price <= 0:
            return None

        return median_price

    # ------------------------------------------------------------------
    def seasonality_multiplier(self, sku_id: str, horizon_days: int) -> float:
        """Estimate seasonal demand uplift/downlift for the forecast horizon."""

        if horizon_days <= 0:
            return 1.0

        day_columns = [col for col in self._sales_columns_list() if col.startswith("d_")]
        if not day_columns:
            return 1.0

        row = self.get_row(sku_id, day_columns)
        if row is None:
            item_id, _ = self._parse_ids(sku_id)
            if item_id:
                row = self.get_row(item_id, day_columns)
        if row is None:
            return 1.0

        values = pd.to_numeric(row[day_columns], errors="coerce").fillna(0.0)
        if values.empty:
            return 1.0

        series = pd.DataFrame({"y": values.to_numpy()})

        start_date = pd.Timestamp("2011-01-29")
        dates = pd.date_range(start=start_date, periods=len(series), freq="D")
        series["month"] = dates.month

        overall_mean = float(series["y"].mean())
        if overall_mean <= 0:
            return 1.0

        month_index = (series.groupby("month")["y"].mean() / overall_mean).to_dict()
        if not month_index:
            return 1.0

        last_date = dates[-1]
        next_months = [
            (last_date + pd.Timedelta(days=offset)).month
            for offset in range(1, horizon_days + 1)
        ]
        multipliers = [month_index.get(month, 1.0) for month in next_months]
        multiplier = float(sum(multipliers) / len(multipliers))

        return max(0.5, min(1.5, multiplier))
