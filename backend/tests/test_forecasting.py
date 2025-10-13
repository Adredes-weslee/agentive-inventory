from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from backend.app.services.forecasting_service import ForecastingService


def _write_m5_stub(tmp_path: Path) -> None:
    sales = pd.DataFrame(
        {
            "id": ["ITEM_1_store"],
            "item_id": ["ITEM_1"],
            "dept_id": ["FOODS"],
            "cat_id": ["FOODS"],
            "store_id": ["CA_1"],
            "state_id": ["CA"],
            **{f"d_{i}": [10 + i % 3] for i in range(1, 11)},
        }
    )
    calendar = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=15, freq="D"),
            "d": [f"d_{i}" for i in range(1, 16)],
        }
    )
    sales.to_csv(tmp_path / "sales_train_validation.csv", index=False)
    calendar.to_csv(tmp_path / "calendar.csv", index=False)


def _write_m5_portfolio_stub(tmp_path: Path) -> None:
    averages = [200.0, 150.0, 90.0, 50.0, 10.0]
    sku_ids = [f"SKU_{idx}" for idx in range(1, 6)]
    item_ids = [f"ITEM_{idx}" for idx in range(1, 6)]

    data = {
        "id": sku_ids,
        "item_id": item_ids,
        "dept_id": ["FOODS"] * len(sku_ids),
        "cat_id": ["FOODS"] * len(sku_ids),
        "store_id": ["CA_1"] * len(sku_ids),
        "state_id": ["CA"] * len(sku_ids),
    }
    for day in range(1, 31):
        data[f"d_{day}"] = averages.copy()

    sales = pd.DataFrame(data)
    calendar = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=60, freq="D"),
            "d": [f"d_{i}" for i in range(1, 61)],
        }
    )

    sales.to_csv(tmp_path / "sales_train_validation.csv", index=False)
    calendar.to_csv(tmp_path / "calendar.csv", index=False)


def test_forecast_returns_expected_horizon(tmp_path: Path) -> None:
    _write_m5_stub(tmp_path)
    service = ForecastingService(data_root=str(tmp_path), moving_average_window=5)

    result = service.forecast("ITEM_1", horizon_days=3)

    assert result.sku_id == "ITEM_1"
    assert result.horizon_days == 3
    assert len(result.forecast) == 3
    dates = [point.date for point in result.forecast]
    assert dates == sorted(dates), "Forecast dates should be strictly increasing"
    assert all(point.model == "sma" for point in result.forecast)
    assert all(point.lo <= point.mean <= point.hi for point in result.forecast)


def test_model_portfolio_assigns_models_by_abc_class(tmp_path: Path) -> None:
    _write_m5_portfolio_stub(tmp_path)
    service = ForecastingService(data_root=str(tmp_path))

    assert service.model_portfolio == {"A": "xgb", "B": "prophet", "C": "sma"}

    assert service._abc_class_for_sku("SKU_1") == "A"
    assert service._abc_class_for_sku("SKU_2") == "B"
    assert service._abc_class_for_sku("SKU_3") == "B"
    assert service._abc_class_for_sku("SKU_4") == "C"

    sku_history_a = service._sku_history("SKU_1")
    sku_history_b = service._sku_history("SKU_2")
    sku_history_c = service._sku_history("SKU_5")

    assert sku_history_a.model_class == "xgb"
    assert sku_history_b.model_class == "prophet"
    assert sku_history_c.model_class == "sma"


def test_backtest_respects_model_hint_override(tmp_path: Path) -> None:
    _write_m5_stub(tmp_path)
    service = ForecastingService(data_root=str(tmp_path), moving_average_window=5)

    result = service.backtest(
        sku_id="ITEM_1",
        window=5,
        horizon=2,
        step=2,
        model_hint="sma",
    )

    assert result["model_used"] == "sma"
