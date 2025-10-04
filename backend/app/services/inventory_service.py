"""Inventory helper utilities built on top of the M5 dataset."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

import pandas as pd

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
        self.data_root = data_root
        self.config_root = config_root or "configs"
        self.sales_df: Optional[pd.DataFrame] = None
        self.prices_df: Optional[pd.DataFrame] = None
        self.settings: Dict[str, Any] = {}
        self._load()
        self._load_settings()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        sales_path = os.path.join(self.data_root, "sales_train_validation.csv")
        prices_path = os.path.join(self.data_root, "sell_prices.csv")

        if os.path.exists(sales_path):
            try:
                self.sales_df = pd.read_csv(sales_path)
            except Exception as exc:  # pragma: no cover - defensive logging only
                LOGGER.warning("Unable to load sales dataset at %s: %s", sales_path, exc)
                self.sales_df = None

        if os.path.exists(prices_path):
            try:
                self.prices_df = pd.read_csv(prices_path)
            except Exception as exc:  # pragma: no cover - defensive logging only
                LOGGER.warning("Unable to load sell prices dataset at %s: %s", prices_path, exc)
                self.prices_df = None

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
    def has_sku(self, sku_id: str) -> bool:
        """Return ``True`` when the SKU exists in the sales dataset."""

        if self.sales_df is None:
            return False

        try:
            return bool((self.sales_df["item_id"] == sku_id).any())
        except KeyError:  # pragma: no cover - defensive logging only
            LOGGER.warning("Sales dataset missing 'item_id' column; cannot confirm SKU %s", sku_id)
            return False

    # ------------------------------------------------------------------
    def sku_exists(self, sku_id: str) -> bool:
        return self.has_sku(sku_id)

    # ------------------------------------------------------------------
    def get_sku_info(self, sku_id: str) -> Dict[str, Any]:
        if self.sales_df is None:
            return {}
        row = self.sales_df[self.sales_df["item_id"] == sku_id]
        if row.empty:
            return {}
        rec = row.iloc[0]
        return {
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

        if self.prices_df is None:
            return None

        sku_prices = self.prices_df[self.prices_df["item_id"] == sku_id]
        if sku_prices.empty:
            return None

        latest = sku_prices.sort_values("wm_yr_wk").iloc[-1]
        return SKUPrice(
            item_id=sku_id,
            store_id=latest.get("store_id"),
            sell_price=float(latest.get("sell_price", 0.0)),
            week=int(latest.get("wm_yr_wk")) if not pd.isna(latest.get("wm_yr_wk")) else None,
        )

    # ------------------------------------------------------------------
    def _get_price_series(self, sku_id: str, store_id: Optional[str] = None) -> Optional[pd.Series]:
        if self.prices_df is None:
            return None

        sku_prices = self.prices_df[self.prices_df["item_id"] == sku_id]
        if sku_prices.empty:
            return pd.Series(dtype=float)

        if store_id:
            store_prices = sku_prices[sku_prices["store_id"] == store_id]
            if not store_prices.empty:
                return store_prices["sell_price"].astype(float)

        return sku_prices["sell_price"].astype(float)

    # ------------------------------------------------------------------
    def get_unit_cost(self, sku_id: str) -> float:
        """Return a proxy unit cost using median sell price data."""

        fallback_cost = 10.0
        store_id = None
        sku_info = self.get_sku_info(sku_id)
        if sku_info:
            store_id = sku_info.get("store_id")

        price_series = self._get_price_series(sku_id, store_id)
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
        store_id = None
        sku_info = self.get_sku_info(sku_id)
        if sku_info:
            store_id = sku_info.get("store_id")

        price_series = self._get_price_series(sku_id, store_id)
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
