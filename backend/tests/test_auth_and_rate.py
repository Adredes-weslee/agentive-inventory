r"""backend/tests/test_auth_and_rate.py"""

from __future__ import annotations

import sys
from collections import defaultdict, deque
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
        demand_cols = {f"d_{i}": [7.0 + (i % 4)] for i in range(1, 61)}
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


def test_auth_and_rate_limit(monkeypatch):
    from backend.app.core import observability as obs

    monkeypatch.setenv("API_TOKEN", "X")
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "1")
    monkeypatch.setattr(obs.TokenAndRateLimitMiddleware, "_token", "X", raising=False)
    monkeypatch.setattr(obs.TokenAndRateLimitMiddleware, "_per_minute", 1, raising=False)
    monkeypatch.setattr(
        obs.TokenAndRateLimitMiddleware,
        "_buckets",
        defaultdict(deque),
        raising=False,
    )

    client = TestClient(app)

    response = client.get("/api/v1/data/validate")
    assert response.status_code == 401

    authed = client.get("/api/v1/data/validate", headers={"Authorization": "Bearer X"})
    assert authed.status_code != 401

    limited = client.get("/api/v1/data/validate", headers={"Authorization": "Bearer X"})
    assert limited.status_code == 429
