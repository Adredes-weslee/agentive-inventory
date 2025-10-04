from __future__ import annotations

import csv
import sys
from datetime import date, timedelta
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

from backend.app.main import app
from backend.app.models import schemas


def _known_sku_id() -> str:
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
    points: list[schemas.ForecastPoint] = []
    for offset in range(horizon_days):
        day = base_day + timedelta(days=offset)
        points.append(
            schemas.ForecastPoint(
                date=day,
                mean=mean,
                lo=max(mean - 1.0, 0.0),
                hi=max(mean + 1.0, 0.0),
                model="sma",
                confidence=0.7,
            )
        )

    class _Result:
        def __init__(self, sku: str, horizon: int, forecast: list[schemas.ForecastPoint]) -> None:
            self.sku_id = sku
            self.horizon_days = horizon
            self.forecast = forecast

    return _Result(sku_id, horizon_days, points)


def _allowlist_inventory(monkeypatch, module_path: str, sku_id: str) -> None:
    def _allowlisted(candidate: str) -> bool:
        return candidate == sku_id

    monkeypatch.setattr(f"{module_path}._inventory_service.has_sku", _allowlisted)
    monkeypatch.setattr(f"{module_path}._inventory_service.sku_exists", _allowlisted)


class _StubProcurementService:
    def recommend(self, forecast: schemas.ForecastResponse, context: dict) -> list[schemas.ReorderRec]:
        means = [float(point.mean) for point in forecast.forecast]
        if not means or sum(means) / len(means) <= 1e-6:
            return []
        return [
            schemas.ReorderRec(
                sku_id=forecast.sku_id,
                reorder_point=5,
                order_qty=10,
                gmroi_delta=1.25,
                confidence=0.8,
                requires_approval=False,
            )
        ]


client = TestClient(app)


def test_procure_api_returns_recommendation_fields(monkeypatch) -> None:
    sku_id = _known_sku_id()
    horizon = 21

    _allowlist_inventory(monkeypatch, "backend.app.api.v1.procure", sku_id)
    def _positive_forecast(sku_id: str, horizon_days: int, **_: object) -> object:
        return _build_stub_forecast(sku_id, horizon_days, mean=18.0)

    monkeypatch.setattr("backend.app.api.v1.procure._forecast_service.forecast", _positive_forecast)
    monkeypatch.setattr("backend.app.api.v1.procure._procurement_service", _StubProcurementService())

    response = client.post(
        "/api/v1/procure/recommendations",
        json={"sku_id": sku_id, "horizon_days": horizon},
    )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload, "Expected non-empty recommendations for positive demand"
    recommendation = payload[0]
    for field in [
        "sku_id",
        "reorder_point",
        "order_qty",
        "gmroi_delta",
        "confidence",
        "requires_approval",
    ]:
        assert field in recommendation


def test_procure_api_returns_empty_for_near_zero_forecast(monkeypatch) -> None:
    sku_id = _known_sku_id()
    horizon = 14

    _allowlist_inventory(monkeypatch, "backend.app.api.v1.procure", sku_id)
    def _zero_forecast(sku_id: str, horizon_days: int, **_: object) -> object:
        return _build_stub_forecast(sku_id, horizon_days, mean=0.0)

    monkeypatch.setattr("backend.app.api.v1.procure._forecast_service.forecast", _zero_forecast)
    monkeypatch.setattr("backend.app.api.v1.procure._procurement_service", _StubProcurementService())

    response = client.post(
        "/api/v1/procure/recommendations",
        json={"sku_id": sku_id, "horizon_days": horizon},
    )

    assert response.status_code == 200
    assert response.json() == []
