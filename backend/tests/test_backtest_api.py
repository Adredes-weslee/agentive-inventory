r"""backend/tests/test_backtest_api.py"""

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
        demand_cols = {f"d_{i}": [10.0 + (i % 5)] for i in range(1, 61)}
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

client = TestClient(app)


def test_backtest_detail(monkeypatch):
    from backend.app.api.v1 import backtest as bt

    monkeypatch.setattr(bt._inventory_service, "has_sku", lambda sku: True)

    fake_response = {
        "dates": ["2020-01-01"],
        "y": [1.0],
        "yhat": [1.1],
        "mape": 0.1,
        "coverage": 1.0,
        "model_used": "sma",
        "origin_dates": ["2019-12-31"],
        "per_origin_coverage": [1.0],
        "history": {"dates": ["2019-12-01"], "y": [0.9]},
    }

    monkeypatch.setattr(
        bt._forecast_service,
        "backtest",
        lambda **kwargs: fake_response,
    )

    response = client.get("/api/v1/backtest/ANY?detail=true")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_used"] == "sma"
    assert payload["history"] == fake_response["history"]
    assert payload["origin_dates"] == fake_response["origin_dates"]
