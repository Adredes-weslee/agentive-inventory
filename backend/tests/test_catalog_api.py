from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

DATA_ROOT = ROOT / "data"
SALES_PATH = (DATA_ROOT / "sales_train_validation.csv").resolve()
CALENDAR_PATH = (DATA_ROOT / "calendar.csv").resolve()
SELL_PRICES_PATH = (DATA_ROOT / "sell_prices.csv").resolve()

FAKE_IDS = [
    "FOODS_1_001_CA_1_validation",
    "HOBBIES_1_002_CA_1_validation",
    "FOODS_3_090_CA_1_validation",
]

_ORIGINAL_READ_CSV = pd.read_csv


def _stubbed_read_csv(path, *args, **kwargs):
    path_obj = Path(path).resolve()
    if path_obj == SALES_PATH:
        demand_cols = {f"d_{day}": [12.0 + day, 8.0 + day, 4.0 + day] for day in range(1, 8)}
        return pd.DataFrame(
            {
                "id": FAKE_IDS,
                "item_id": ["FOODS_1_001", "HOBBIES_1_002", "FOODS_3_090"],
                "dept_id": ["FOODS_1", "HOBBIES_1", "FOODS_3"],
                "cat_id": ["FOODS", "HOBBIES", "FOODS"],
                "store_id": ["CA_1", "CA_1", "CA_1"],
                "state_id": ["CA", "CA", "CA"],
                **demand_cols,
            }
        )
    if path_obj == CALENDAR_PATH:
        dates = pd.date_range("2020-01-01", periods=400, freq="D")
        return pd.DataFrame({"date": dates, "d": [f"d_{i}" for i in range(1, 401)]})
    if path_obj == SELL_PRICES_PATH:
        return pd.DataFrame(
            {
                "item_id": ["FOODS_1_001", "HOBBIES_1_002", "FOODS_3_090"],
                "store_id": ["CA_1", "CA_1", "CA_1"],
                "wm_yr_wk": [11101, 11102, 11103],
                "sell_price": [4.99, 7.49, 3.25],
            }
        )
    return _ORIGINAL_READ_CSV(path, *args, **kwargs)


pd.read_csv = _stubbed_read_csv  # type: ignore[assignment]

from backend.app.main import app  # noqa: E402


client = TestClient(app)


def test_catalog_ids_works() -> None:
    response = client.get("/api/v1/catalog/ids", params={"limit": 2})
    assert response.status_code == 200
    payload = response.json()
    assert "ids" in payload
    ids = payload["ids"]
    assert isinstance(ids, list)
    assert ids == FAKE_IDS[:2]
