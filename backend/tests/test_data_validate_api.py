r"""backend/tests/test_data_validate_api.py"""

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
        demand_cols = {f"d_{i}": [5.0 + (i % 2)] for i in range(1, 61)}
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


def test_validate_ok(monkeypatch, tmp_path: Path):
    sales = tmp_path / "sales_train_validation.csv"
    prices = tmp_path / "sell_prices.csv"
    calendar = tmp_path / "calendar.csv"

    sales.write_text("id,item_id,d_1,d_2\nA,B,1,2\n", encoding="utf-8")
    prices.write_text("item_id,wm_yr_wk,sell_price\nB,1,2.0\n", encoding="utf-8")
    calendar.write_text(
        "date,d,snap_CA,snap_TX,snap_WI\n2016-01-01,d_1,0,0,0\n2016-01-02,d_2,1,0,0\n",
        encoding="utf-8",
    )

    from backend.app.api.v1 import data as data_api

    monkeypatch.setattr(data_api._validation_service, "data_root", str(tmp_path))

    response = client.get("/api/v1/data/validate")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["checks"], list)
    assert payload["checks"]
