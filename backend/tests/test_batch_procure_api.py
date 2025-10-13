from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

DATA_ROOT = ROOT / "data"
SALES_PATH = DATA_ROOT / "sales_train_validation.csv"
CALENDAR_PATH = DATA_ROOT / "calendar.csv"
SELL_PRICES_PATH = DATA_ROOT / "sell_prices.csv"

_ORIGINAL_READ_CSV = pd.read_csv


def _stubbed_read_csv(path, *args, **kwargs):
    path_obj = Path(path).resolve()
    if path_obj == SALES_PATH:
        demand_cols = {f"d_{i}": [8.0 + (i % 3)] for i in range(1, 61)}
        return pd.DataFrame(
            {
                "id": ["HOBBIES_1_001_CA_1_validation"],
                "item_id": ["HOBBIES_1_001"],
                "dept_id": ["HOBBIES_1"],
                "cat_id": ["HOBBIES"],
                "store_id": ["CA_1"],
                "state_id": ["CA"],
                **demand_cols,
            }
        )
    if path_obj == CALENDAR_PATH:
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        return pd.DataFrame({"date": dates, "d": [f"d_{i}" for i in range(1, 201)]})
    if path_obj == SELL_PRICES_PATH:
        return pd.DataFrame(
            {
                "item_id": ["HOBBIES_1_001"],
                "store_id": ["CA_1"],
                "wm_yr_wk": [11101],
                "sell_price": [9.99],
            }
        )
    return _ORIGINAL_READ_CSV(path, *args, **kwargs)


pd.read_csv = _stubbed_read_csv  # type: ignore[assignment]

from backend.app.main import app  # noqa: E402
from backend.app.models import schemas  # noqa: E402

client = TestClient(app)


def test_batch_selection(monkeypatch):
    from backend.app.api.v1 import procure as pr

    def fake_generate(sku_id: str, horizon: int):
        order_qty = 10 if sku_id.endswith("A") else 5
        gmroi = 0.5 if sku_id.endswith("A") else 0.2
        return [
            schemas.ReorderRec(
                sku_id=sku_id,
                reorder_point=1,
                order_qty=order_qty,
                gmroi_delta=gmroi,
                confidence=0.9,
                requires_approval=False,
            )
        ]

    monkeypatch.setattr(pr, "_generate_recommendations", fake_generate)
    monkeypatch.setattr(pr._inventory_service, "get_unit_cost", lambda sku: 10.0)

    body = {"sku_ids": ["ITEMA", "ITEMB"], "horizon_days": 28, "cash_budget": 60.0}
    response = client.post("/api/v1/procure/batch_recommendations", json=body)
    assert response.status_code == 200
    payload = response.json()
    selected = [rec for rec in payload["recommendations"] if rec["selected"]]
    assert len(selected) == 1
    assert selected[0]["sku_id"] == "ITEMB"
    assert payload["total_spend_selected"] == selected[0]["total_spend"]
