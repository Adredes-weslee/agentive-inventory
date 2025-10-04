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
