"""
Service to manage inventory state and SKU metadata.

This service provides helper methods to look up SKU information and
inventory levels.  In a real system this might connect to an ERP or
database; here we derive some information from the M5 dataset when
available.  If the dataset is not present, stub values are returned.
"""

from __future__ import annotations

import os
import pandas as pd
from typing import Optional, Dict, Any


class InventoryService:
    def __init__(self, data_root: str = "data") -> None:
        self.data_root = data_root
        self.sales_df: Optional[pd.DataFrame] = None
        self._load()

    def _load(self) -> None:
        """Attempt to load the sales dataset to extract SKU metadata."""
        path = os.path.join(self.data_root, "sales_train_validation.csv")
        if os.path.exists(path):
            try:
                self.sales_df = pd.read_csv(path)
            except Exception:
                self.sales_df = None

    def get_sku_info(self, sku_id: str) -> Dict[str, Any]:
        """Return basic metadata for a SKU.

        Currently this returns the department and category identifiers from
        the sales dataset if available.  Otherwise it returns an empty
        dictionary.
        """
        if self.sales_df is None:
            return {}
        row = self.sales_df[self.sales_df["item_id"] == sku_id]
        if row.empty:
            return {}
        # Take the first matching record
        rec = row.iloc[0]
        return {
            "dept_id": rec.get("dept_id"),
            "cat_id": rec.get("cat_id"),
            "store_id": rec.get("store_id"),
        }

    def get_current_inventory(self, sku_id: str) -> int:
        """Return the current on‑hand inventory level for a SKU.

        In a real system this would query an inventory database; here it
        always returns zero because we don't track inventory balances.
        """
        return 0