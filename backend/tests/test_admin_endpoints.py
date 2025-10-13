from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd
import yaml
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
        demand_cols = {f"d_{i}": [5.0 + (i % 2)] for i in range(1, 31)}
        return pd.DataFrame(
            {
                "id": ["FOODS_1_001_CA_1_validation"],
                "item_id": ["FOODS_1_001"],
                "dept_id": ["FOODS_1"],
                "cat_id": ["FOODS"],
                "store_id": ["CA_1"],
                "state_id": ["CA"],
                **demand_cols,
            }
        )
    if path_obj == CALENDAR_PATH:
        dates = pd.date_range("2021-01-01", periods=200, freq="D")
        return pd.DataFrame({"date": dates, "d": [f"d_{i}" for i in range(1, 201)]})
    if path_obj == SELL_PRICES_PATH:
        return pd.DataFrame(
            {
                "item_id": ["FOODS_1_001"],
                "store_id": ["CA_1"],
                "wm_yr_wk": [11101],
                "sell_price": [9.99],
            }
        )
    return _ORIGINAL_READ_CSV(path, *args, **kwargs)


pd.read_csv = _stubbed_read_csv  # type: ignore[assignment]


from backend.app.main import app  # noqa: E402


client = TestClient(app)


def test_configs_get_put(monkeypatch, tmp_path: Path) -> None:
    cfg_module = "backend.app.api.v1.configs"
    monkeypatch.setattr(f"{cfg_module}.CONFIG_DIR", str(tmp_path))
    (tmp_path / "settings.yaml").write_text(
        yaml.safe_dump({"service_level_target": 0.95, "lead_time_days": 14})
    )
    (tmp_path / "thresholds.yaml").write_text(
        yaml.safe_dump({"auto_approval_limit": 500.0})
    )

    response = client.get("/api/v1/configs/settings")
    assert response.status_code == 200

    response = client.put("/api/v1/configs/settings", json={"lead_time_days": 10})
    assert response.status_code == 200
    settings = yaml.safe_load((tmp_path / "settings.yaml").read_text())
    assert settings["lead_time_days"] == 10

    response = client.get("/api/v1/configs/thresholds")
    assert response.status_code == 200

    response = client.put(
        "/api/v1/configs/thresholds",
        json={"auto_approval_limit": 750.0},
    )
    assert response.status_code == 200
    thresholds = yaml.safe_load((tmp_path / "thresholds.yaml").read_text())
    assert thresholds["auto_approval_limit"] == 750.0


def test_approvals_audit_log(monkeypatch, tmp_path: Path) -> None:
    appr_module = "backend.app.api.v1.approvals"
    monkeypatch.setattr(f"{appr_module}.DATA_DIR", str(tmp_path))
    monkeypatch.setattr(f"{appr_module}.LOG_PATH", str(tmp_path / "audit_log.jsonl"))

    response = client.post(
        "/api/v1/approvals",
        json={"sku_id": "ITEM_1", "action": "approve", "qty": 5, "reason": "pilot"},
    )
    assert response.status_code == 200

    response = client.get("/api/v1/approvals/audit-log", params={"limit": 10})
    assert response.status_code == 200
    assert len(response.json()["events"]) >= 1


def test_explain_returns_404_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    body = {"sku_id": "ITEM_1", "horizon_days": 7, "recommendations": []}

    response = client.post("/api/v1/procure/recommendations/explain", json=body)
    assert response.status_code == 404


def test_explain_works_with_stubbed_llm(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    cfg_module = "backend.app.api.v1.procure"

    (tmp_path / "settings.yaml").write_text(yaml.safe_dump({"service_level_target": 0.9}))
    (tmp_path / "thresholds.yaml").write_text(yaml.safe_dump({"auto_approval_limit": 1000.0}))

    def _load(name: str) -> dict:
        path = tmp_path / name
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
        return {}

    monkeypatch.setattr(f"{cfg_module}._load_yaml_config", _load)
    monkeypatch.setattr(
        "backend.app.api.v1.procure.explain_recommendation",
        lambda ctx, recs: "stub explanation",
    )

    body = {"sku_id": "ITEM_1", "horizon_days": 7, "recommendations": []}

    response = client.post("/api/v1/procure/recommendations/explain", json=body)
    assert response.status_code == 200
    assert response.json()["explanation"].startswith("stub")
