"""Inventory helper utilities built on top of the M5 dataset."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

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

    def __init__(self, data_root: str = "data") -> None:
        self.data_root = data_root
        self.sales_df: Optional[pd.DataFrame] = None
        self.prices_df: Optional[pd.DataFrame] = None
        self._load()

    # ------------------------------------------------------------------
    def get_unit_cost(self, sku_id: str) -> float:
        """Return an estimated unit cost for ``sku_id``."""

        return float(self.estimate_unit_cost(sku_id))

    # ------------------------------------------------------------------
    def get_lead_time_days(self, sku_id: str) -> float:
        """Return the best-known replenishment lead time in days."""

        # The reference datasets do not expose lead-time information.  We
        # therefore use a conservative 7-day placeholder which downstream
        # services can override via context or configuration.
        _ = sku_id  # Intentionally unused but keeps signature explicit.
        return 7.0

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
    def sku_exists(self, sku_id: str) -> bool:
        return bool(self.sales_df is not None and not self.sales_df[self.sales_df["item_id"] == sku_id].empty)

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
        """Return stub on‑hand inventory levels.

        In production this would connect to an inventory store.  For the M5
        dataset we do not have stock balances so a conservative placeholder
        of ``0`` is returned.
        """

        return 0

    # ------------------------------------------------------------------
    @lru_cache(maxsize=1024)
    def get_latest_price(self, sku_id: str) -> Optional[SKUPrice]:
        """Return the most recent sell price for the SKU.

        Sell prices in the M5 dataset are weekly and store specific.  We take
        the latest known price across all stores and return it as a proxy for
        unit cost.
        """

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
    def estimate_unit_cost(self, sku_id: str) -> float:
        """Estimate the per‑unit acquisition cost for a SKU.

        Without procurement cost data we fall back to the latest known sell
        price as a proxy.  If pricing data is missing we return ``1.0`` to
        keep downstream economic calculations finite.
        """

        price = self.get_latest_price(sku_id)
        if price is None or price.sell_price <= 0:
            return 1.0
        # Treat sell price as a margin‑inclusive figure; assume a modest 25%
        # gross margin to back into an approximate unit cost.
        assumed_margin = 0.25
        unit_cost = price.sell_price * (1 - assumed_margin)
        return max(unit_cost, 0.5)