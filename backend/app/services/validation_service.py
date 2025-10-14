r"""backend\app\services\validation_service.py"""

from __future__ import annotations

import os
import pandas as pd

REQUIRED_SALES_COLS = ["id", "item_id"]
REQUIRED_PRICE_COLS = ["item_id", "wm_yr_wk", "sell_price"]


class ValidationService:
    def __init__(self, data_root: str | None = None):
        self.data_root = data_root or os.getenv("DATA_DIR", "data")

    def run(self) -> dict:
        checks = []

        def add(name: str, ok: bool, msg: str = "") -> None:
            checks.append({"name": name, "ok": bool(ok), "message": msg})

        sales = os.path.join(self.data_root, "sales_train_validation.csv")
        cal = os.path.join(self.data_root, "calendar.csv")
        prices = os.path.join(self.data_root, "sell_prices.csv")

        add("file_sales_exists", os.path.exists(sales), sales)
        add("file_calendar_exists", os.path.exists(cal), cal)
        add("file_prices_exists", os.path.exists(prices), prices)

        if os.path.exists(sales):
            df = pd.read_csv(sales, nrows=3)
            ok = all(c in df.columns for c in REQUIRED_SALES_COLS) and any(
                c.startswith("d_") for c in df.columns
            )
            add("sales_columns_ok", ok, f"have: {list(df.columns)[:8]}...")

        if os.path.exists(prices):
            dfp = pd.read_csv(prices, nrows=3)
            ok = all(c in dfp.columns for c in REQUIRED_PRICE_COLS)
            add("prices_columns_ok", ok, f"have: {list(dfp.columns)[:8]}...")

        overall = all(x["ok"] for x in checks)
        return {"ok": overall, "checks": checks}
