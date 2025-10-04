from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.app.models.schemas import ForecastPoint, ForecastResponse
from backend.app.services.procurement_service import ProcurementService


class StubInventoryService:
    def estimate_unit_cost(self, sku_id: str) -> float:  # pragma: no cover - trivial
        return 25.0


def _write_configs(tmp_path: Path) -> Path:
    config_root = tmp_path / "cfg"
    config_root.mkdir()
    thresholds = {
        "auto_approval_limit": 100.0,
        "min_service_level": 0.85,
        "gmroi_min": 1.0,
        "max_cash_outlay": 10_000.0,
    }
    settings = {
        "carrying_cost_rate": 0.25,
        "service_level_target": 0.9,
        "lead_time_days": 7,
        "order_cost": 50.0,
        "gross_margin_rate": 0.3,
    }
    (config_root / "thresholds.yaml").write_text(yaml.safe_dump(thresholds))
    (config_root / "settings.yaml").write_text(yaml.safe_dump(settings))
    return config_root


def _build_forecast(horizon: int = 14) -> ForecastResponse:
    start = date(2020, 1, 1)
    points = []
    for i in range(horizon):
        day = start + timedelta(days=i)
        points.append(
            ForecastPoint(
                date=day,
                mean=20.0,
                lo=16.0,
                hi=24.0,
                model="sma",
                confidence=0.7,
            )
        )
    return ForecastResponse(sku_id="ITEM_1", horizon_days=horizon, forecast=points)


def test_requires_approval_when_spend_exceeds_limit(tmp_path: Path) -> None:
    config_root = _write_configs(tmp_path)
    service = ProcurementService(config_root=str(config_root), inventory_service=StubInventoryService())
    forecast = _build_forecast()

    recs = service.recommend(forecast, context={})

    assert len(recs) == 1
    recommendation = recs[0]
    assert recommendation.order_qty > 0
    assert recommendation.requires_approval is True
    assert recommendation.confidence <= 0.6
