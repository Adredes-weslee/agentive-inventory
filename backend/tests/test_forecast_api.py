r"""backend/tests/test_forecast_api.py"""

from __future__ import annotations

import csv
import sys
from datetime import date, datetime, timedelta
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
FALLBACK_SKU = "HOBBIES_1_001"

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

from backend.app.main import app
from backend.app.models import schemas


def _known_sku_id() -> str:
    """Return a SKU identifier from the sales dataset or fall back to a default."""

    try:
        with SALES_PATH.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if "item_id" not in fieldnames:
                raise ValueError("sales dataset missing item_id column")
            first_row = next(reader)
            if first_row and first_row.get("item_id"):
                return str(first_row["item_id"])
    except Exception:
        pass
    return FALLBACK_SKU


def _build_stub_forecast(sku_id: str, horizon_days: int, mean: float) -> object:
    base_day = date.today()
    forecast_points = []
    for offset in range(horizon_days):
        day = base_day + timedelta(days=offset)
        forecast_points.append(
            schemas.ForecastPoint(
                date=day,
                mean=mean,
                lo=max(mean - 1.0, 0.0),
                hi=max(mean + 1.0, 0.0),
                model="sma",
                confidence=0.75,
            )
        )

    class _Result:
        def __init__(self, sku: str, horizon: int, points: list[schemas.ForecastPoint]) -> None:
            self.sku_id = sku
            self.horizon_days = horizon
            self.forecast = points

    return _Result(sku_id, horizon_days, forecast_points)


client = TestClient(app)


def test_forecast_api_returns_expected_payload(monkeypatch) -> None:
    sku_id = _known_sku_id()
    horizon = 14

    def _allowlisted_sku(candidate: str) -> bool:
        return candidate == sku_id

    monkeypatch.setattr(
        "backend.app.api.v1.forecasts._inventory_service.has_sku",
        _allowlisted_sku,
    )
    monkeypatch.setattr(
        "backend.app.api.v1.forecasts._inventory_service.sku_exists",
        _allowlisted_sku,
    )
    def _fake_forecast(sku_id: str, horizon_days: int, **_: object) -> object:
        return _build_stub_forecast(sku_id, horizon_days, mean=12.0)

    monkeypatch.setattr("backend.app.api.v1.forecasts._forecast_service.forecast", _fake_forecast)

    response = client.get(f"/api/v1/forecasts/{sku_id}", params={"horizon_days": horizon})

    assert response.status_code == 200
    payload = response.json()
    assert payload["sku_id"] == sku_id
    assert payload["horizon_days"] == horizon
    assert len(payload["forecast"]) == horizon
    for point in payload["forecast"]:
        # ``fromisoformat`` raises if the string is not ISO-8601 compliant.
        datetime.fromisoformat(point["date"])
        assert point["model"] in {"sma", "prophet", "xgb"}
