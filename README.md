# Agentive Inventory Management System

Agentic, human-in-the-loop inventory planning. This repo provides:

* **FastAPI** backend for forecasts & procurement (EOQ/ROP + guardrails)
* **Streamlit** UI (“Human Collaboration Hub”)
* **Configs** for business context & thresholds
* **n8n** workflow example (daily ingest → forecast → recommend → approve)

It uses the **M5 Forecasting** dataset (Walmart) as the canonical demand history.

---

## Repo layout

```
agentive-inventory/
├─ README.md
├─ .env.example
├─ pyproject.toml
├─ configs/
│  ├─ settings.yaml
│  └─ thresholds.yaml
├─ data/                     # Put M5 CSVs here (not committed)
│  ├─ sales_train_validation.csv
│  ├─ calendar.csv
│  └─ sell_prices.csv
├─ backend/
│  ├─ app/
│  │  ├─ main.py
│  │  ├─ api/v1/{forecasts.py, procure.py, health.py}
│  │  ├─ core/config.py
│  │  ├─ models/schemas.py
│  │  └─ services/{forecasting_service.py, procurement_service.py, inventory_service.py, llm_service.py}
│  ├─ requirements.txt
│  └─ Dockerfile
├─ frontend/
│  ├─ app.py
│  ├─ pages/{1_Dashboard.py, 2_Forecasts.py, 3_Recommendations.py, 4_Settings.py}
│  └─ requirements.txt
├─ orchestration/n8n_workflows/example_workflow.json
├─ infra/{docker-compose.yml, render.yaml, k8s/}
└─ .github/workflows/ci.yml
```

---

## Data (M5)

Download from the official **M5 Forecasting – Accuracy** competition page and place the CSVs in `./data/`. Files:

* `sales_train_validation.csv`
* `calendar.csv`
* `sell_prices.csv`

These are **not** included in the repo.

---

## Quick start (local)

### Windows (PowerShell)

```powershell
# 1) Clone
git clone <your-repo-url>.git
cd agentive-inventory

# 2) Place M5 CSVs into .\data\
#    sales_train_validation.csv, calendar.csv, sell_prices.csv

# 3) Backend
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload

# 4) Frontend (new terminal)
pip install -r requirements.txt
$env:API_URL = "http://localhost:8000/api/v1"
python -m streamlit run frontend/app.py
```

---

## API smoke tests

**Find a real M5 SKU id** (first row id):

*PowerShell*

```powershell
Get-Content .\data\sales_train_validation.csv -TotalCount 2 | Select-Object -Last 1
# copy the first field up to the first comma (that's the id)
```


**Call the API** (replace `<SKU_ID>`):

*PowerShell*

```powershell
Invoke-RestMethod "http://localhost:8000/api/v1/forecasts/<SKU_ID>?horizon_days=28"
Invoke-RestMethod -Method Post "http://localhost:8000/api/v1/procure/recommendations" `
  -Body (@{ sku_id = "<SKU_ID>"; horizon_days = 28 } | ConvertTo-Json) -ContentType "application/json"
```


Expected:

* `/forecasts` returns `horizon_days` points with fields: `date, mean, lo, hi, model, confidence`.
* `/procure/recommendations` returns a list with `order_qty, reorder_point, gmroi_delta, confidence, requires_approval`.

---

## Test & debug guide

### Quick tests

```bash
pip install pytest
pytest backend/tests -q
```

Notes:

* Tests **skip** if M5 CSVs are missing.
* If a test fails, read the error for the missing file or shape mismatch.

### Common issues → fixes

* **File not found (calendar or sales CSVs)**
  Put the three M5 CSVs into `./data/` and restart the backend.
* **404 “SKU not found”**
  Use a valid id from the first column of `sales_train_validation.csv`.
* **CORS / UI can’t reach API**
  Ensure `API_URL` is set in the UI environment to your API origin (localhost or Render URL).
  If deploying, allow CORS origins in `backend/app/main.py` via `CORS_ORIGINS`.
* **Port already in use**
  Change the port: `uvicorn app.main:app --app-dir backend/app --port 8001`.
* **Slow first call**
  The service caches calendar and sales header columns after the first hit; warm up by calling `/api/v1/health`, then `/forecasts`.
* **Procurement `requires_approval` always true/false**
  Tune `configs/thresholds.yaml` (`auto_approval_limit, min_service_level, gmroi_min`) and `configs/settings.yaml` (`service_level_target, carrying_cost_rate, lead_time_days`).

### Logging tips

* Start uvicorn with access logs and debug:

  ```bash
  python -m uvicorn backend.app.main:app --reload --log-level debug
  ```
* Add prints/logs where needed in `services/*` (e.g., chosen model, EOQ/ROP inputs).

### Sanity checks

* Choose 2–3 SKUs with different volumes; confirm:

  * Forecast dates are contiguous and length matches `horizon_days`.
  * Procurement returns a positive `order_qty` and plausible `reorder_point`.
  * Flipping thresholds changes `requires_approval` as expected.

---

## Runbook

### Backtesting
1. Open the **🧪 Backtest** page in the Streamlit UI.
2. Enter a valid M5 row id, choose the `window`, `horizon`, and `step` parameters, then submit.
3. Select a model (`auto`, `sma`, `prophet`, or `xgb`). The API returns metrics (MAPE, prediction interval coverage) and offers CSV export.

### Approvals & audit log
1. Visit the **3_Recommendations** page to review pending rows, then click **Approve** or **Reject** with an optional note.
2. Switch to **🗒️ Audit Log** to view decisions. On Render, entries persist when the API service has a disk attached.

### Render deployment checklist
* The API service defined in `infra/render.yaml` should mount a persistent disk at `/data`; the audit log (`audit_log.jsonl`) will be stored there across deploys.
* Set `API_TOKEN` on the API service and configure the Streamlit UI to send that token for protected endpoints.
* Restrict cross-origin access by setting `CORS_ORIGINS` to your Streamlit URL (e.g., `https://<your-ui>.onrender.com`).
* Apply rate limiting at the edge (Render firewall or CDN) and rotate tokens regularly to minimize risk of abuse.

---

## Configuration quick reference

* `configs/settings.yaml` → `service_level_target`, `lead_time_days`, `carrying_cost_rate`, `order_setup_cost`
* `configs/thresholds.yaml` → `auto_approval_limit`, `min_service_level`, `gmroi_min`
* `.env` (copy from `.env.example`) → optional `GEMINI_API_KEY`, `API_PORT`, etc.

---

## Docker / Compose (local)

```bash
docker compose up --build
```

Exposes API on `http://localhost:8000` and UI on `http://localhost:8501`.
Mount `./data` to make M5 files visible inside the containers (see `infra/docker-compose.yml`).

---

## Deploy (Render)

Two services in **Render**:

* **agentive-api** (rootDir=`backend`)
  Build: `pip install -r requirements.txt`
  Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
  Env: `GEMINI_API_KEY` (optional), `DATA_DIR=/data` (keep this pointing to the Render disk mount).
  Disk: attach a persistent disk (e.g., 2 GB) named `agentive-data` mounted at `/data` for saved files.
* **agentive-ui** (rootDir=`frontend`)
  Build: `pip install -r requirements.txt`
  Start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
  Env: `API_URL=https://<your-api>.onrender.com/api/v1`

---

## Roadmap (optional)

* Add Prophet/XGBoost model switch + backtesting route
* Improve GMROI proxy and add cash budget constraints
* Streamlit approvals & audit log
* n8n scheduled daily flow (import `orchestration/n8n_workflows/example_workflow.json`)
