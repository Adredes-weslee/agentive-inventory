# Agentive Inventory Management System

Agentic, human-in-the-loop inventory planning.

This repo ships:

* **FastAPI** backend for forecasts & procurement (EOQ/ROP + guardrails)
* **Streamlit** UI (“Human Collaboration Hub”)
* **Configs** for business context & thresholds
* **n8n** workflows (daily run + approval loop)
* **Gemini (optional)** natural-language rationales for recommendations
* **CI (ruff + mypy + pytest)**, Docker Compose, and Render deploys
* **Optional API auth + rate-limit**, structured JSON logs, and **/metrics** (Prometheus format)

Uses the **M5 Forecasting – Accuracy** dataset (Walmart) as canonical demand history.

---

## Repository layout

```
agentive-inventory/
├─ README.md
├─ .env.example
├─ pyproject.toml
├─ configs/
│  ├─ settings.yaml
│  └─ thresholds.yaml
├─ data/                         # Put M5 CSVs here
│  ├─ sales_train_validation.csv
│  ├─ calendar.csv
│  └─ sell_prices.csv
├─ backend/
│  ├─ app/
│  │  ├─ main.py
│  │  ├─ api/v1/
│  │  │  ├─ health.py                    # GET /health
│  │  │  ├─ forecasts.py                 # GET /forecasts/{sku_id}
│  │  │  ├─ procure.py                   # POST /procure/*
│  │  │  ├─ catalog.py                   # GET /catalog/ids
│  │  │  ├─ configs.py                   # GET/PUT /configs/{settings,thresholds}
│  │  │  ├─ approvals.py                 # POST /approvals; GET /approvals/audit-log
│  │  │  └─ backtest.py                  # GET /backtest/{sku_id}
│  │  ├─ core/
│  │  │  ├─ config.py
│  │  │  └─ observability.py             # auth, rate-limit, logs, /metrics
│  │  ├─ models/schemas.py
│  │  └─ services/
│  │     ├─ forecasting_service.py       # SMA/Prophet/XGBoost + backtests + joblib cache
│  │     ├─ procurement_service.py       # EOQ/ROP + GMROI guardrails (+ seasonality)
│  │     ├─ inventory_service.py         # M5 loaders, unit/price lookup, seasonality
│  │     └─ llm_service.py               # Gemini explanations (optional)
│  ├─ requirements.txt
│  └─ Dockerfile
├─ frontend/
│  ├─ app.py                            # Health banner + API token sidebar
│  └─ pages/
│     ├─ 1_Dashboard.py                 # KPIs + PI band + CSV export
│     ├─ 2_Forecasts.py                 # Typeahead (GET /catalog/ids), CSV export
│     ├─ 3_Recommendations.py           # Explain, Approve/Reject, Batch mode
│     ├─ 4_Settings.py                  # Edit + PUT /configs/{settings,thresholds}
│     ├─ 5_Backtest.py                  # Rolling backtest + history overlay
│     └─ 6_Audit_Log.py                 # Read /approvals/audit-log
├─ orchestration/
│  ├─ README.md
│  └─ n8n_workflows/
│     ├─ example_workflow.json          # Daily schedule → forecast → recommend
│     └─ approval_loop.json             # Human decision → POST /approvals
├─ infra/
│  └─ render.yaml                       # two Render services (API + UI)
├─ docker-compose.yml                   # root compose (uses ./.env)
└─ .github/workflows/ci.yml             # ruff + mypy + pytest + coverage
```
---

## Data (M5)

Download from the **M5 Forecasting – Accuracy** competition and place these CSVs in `./data/`:

* `sales_train_validation.csv`
* `calendar.csv`
* `sell_prices.csv`

These are committed.

---

## Quick start (local)

### One-command (Docker Compose)

```bash
# 1) Clone
git clone <your-repo-url>.git
cd agentive-inventory

# 2) Put M5 CSVs into ./data/
#    sales_train_validation.csv, calendar.csv, sell_prices.csv

# 3) (Optional) copy env and edit
cp .env.example .env

# 4) Bring everything up (API :8000, UI :8501, n8n :5678)
docker compose up --build
```

Open:

* UI → [http://localhost:8501](http://localhost:8501)
* API docs → [http://localhost:8000/docs](http://localhost:8000/docs)
* Metrics → [http://localhost:8000/metrics](http://localhost:8000/metrics)
* n8n → [http://localhost:5678](http://localhost:5678)

### Python (conda, no containers)

```powershell
# Create and activate env (once)
conda env create -f environment.yml
conda activate agentive-inventory
```

Backend (new PowerShell)
```powershell
conda activate agentive-inventory
# Optional: load settings from .env automatically
# Also set DATA_DIR/CONFIG_DIR for file paths
$env:DATA_DIR = ".\data"
$env:CONFIG_DIR = ".\configs"

# If using auth, ensure the token matches your .env (API_TOKEN=dev-12345 by default)
# $env:API_TOKEN = "dev-12345"

python -m uvicorn backend.app.main:app --reload --port 8000 --env-file .env
```

Frontend (another PowerShell)
```powershell
conda activate agentive-inventory
$env:API_URL = "http://localhost:8000/api/v1"

# If API auth is enabled, provide the same token (or paste it in the UI sidebar)
# $env:API_TOKEN = "dev-12345"

python -m streamlit run frontend/app.py
```

---

## UI overview

* **Dashboard**: KPIs (avg demand, 28-day total, avg PI width), mean + PI band chart, CSV export.
* **Forecasts**: Typeahead of valid M5 row **`id`** via `/catalog/ids`, caching, CSV export.
* **Recommendations**: `/procure/recommendations` → recommended order qty + guardrails.
  Explain via `/procure/recommendations/explain` (Gemini; falls back to a simple heuristic text if configured), Approve/Reject via `/approvals`. Batch tab supports multiselect/pasted lists and optional cash budget.
* **Settings**: Edit & persist via `PUT /configs/settings` & `PUT /configs/thresholds`.
* **Backtest**: Rolling-origin metrics via `/backtest` with **history overlay** and model selection.
* **Audit Log**: Stream of approval events from `/approvals/audit-log`.
* **Sidebar**: API token box; stored in session and sent as `Authorization: Bearer <token>`.

---

## API highlights

* **Forecasts**

  * `GET /api/v1/forecasts/{sku_id}?horizon_days=28`
* **Backtesting**

  * `GET /api/v1/backtest/{sku_id}?window=56&horizon=28&step=7&model=auto`
  * Returns arrays (`dates`, `y`, `yhat`), summary metrics (`mape`, `coverage`, `model_used`), per-origin coverage, and recent history (`history_dates`, `history_values`).
* **Catalog**

  * `GET /api/v1/catalog/ids?limit=20` → sample M5 row ids for typeahead
* **Procurement**

  * `POST /api/v1/procure/recommendations`
  * `POST /api/v1/procure/recommendations/explain` → rationale (200 when enabled; **404 when disabled by design**)
  * `POST /api/v1/procure/batch_recommendations` → batch selection under optional cash budget
* **Configs**

  * `GET/PUT /api/v1/configs/settings`
  * `GET/PUT /api/v1/configs/thresholds`
* **Approvals**

  * `POST /api/v1/approvals` `{sku_id, action: approve|reject, qty, reason}`
  * `GET /api/v1/approvals/audit-log?limit=100`
* **Health / Metrics**

  * `GET /api/v1/health`
  * `GET /metrics` (Prometheus text; works with or without `prometheus_client` thanks to a built-in fallback)

**Response shapes** (abbrev):

* `/forecasts/{sku}` → `{ sku_id, horizon_days, forecast: [{date, mean, lo, hi, model, confidence}] }`
* `/procure/recommendations` → `[{ sku_id, order_qty, reorder_point, gmroi_delta, confidence, requires_approval }]`
* `/backtest/{sku}` → `{ dates, y, yhat, mape, coverage, per_origin_coverage, history_dates, history_values, model_used }`

---

## Configuration quick reference

* `configs/settings.yaml`

  * `service_level_target`, `lead_time_days`, `carrying_cost_rate`, `order_setup_cost`, `default_unit_cost`, `gross_margin_rate`
* `configs/thresholds.yaml`

  * `auto_approval_limit`, `min_service_level`, `gmroi_min`
* `.env` (copy from `.env.example`)

  * API: `API_HOST`, `API_PORT`, `DATA_DIR`, `CONFIG_DIR`, `CORS_ORIGINS`
  * LLM (optional): `GEMINI_API_KEY` (omit to disable explanations), `GEMINI_MODEL` (e.g. `gemini-1.5-flash`)
  * Security/limits: `API_TOKEN` (enable auth), `RATE_LIMIT_PER_MIN` (e.g., `60`)

> **Explanation endpoint behavior**
> If `GEMINI_API_KEY` is **unset**, `/procure/recommendations/explain` returns **404** (expected; tests rely on this).
> You can opt-in to always return a heuristic explanation with `EXPLAIN_FALLBACK_WHEN_DISABLED=true`.

**Model cache** lives in **`/data/models`** (joblib). Approval audit lives at **`data/audit_log.jsonl`**.

---

## Security & observability (optional)

* **Auth**: Set `API_TOKEN` on the API. The UI sidebar sends `Authorization: Bearer …`.
* **Rate limit**: `RATE_LIMIT_PER_MIN` per-IP sliding window.
* **CORS**: Restrict `CORS_ORIGINS` to your UI origin(s).
* **Logs**: Structured JSON per request (`request_id`, `sku_id` (now extracted from POST bodies on known routes), `model_used`, latency, status).
* **Metrics**: `/metrics` serves Prometheus text; a lightweight fallback is used if `prometheus_client` isn’t installed.

---

## Orchestration (n8n)

* Import `orchestration/n8n_workflows/example_workflow.json` (daily schedule → forecast → recommend).
* Import `orchestration/n8n_workflows/approval_loop.json` (notify approver via Slack/email, receive decision via webhook, then `POST /approvals`).
* Set `API_URL` in n8n env (Compose preset: `http://backend:8000/api/v1`). Add header `Authorization: Bearer <token>` if using auth.

---

## CI, tests & quality

* **CI**: `.github/workflows/ci.yml` runs ruff, mypy, and pytest with coverage.
* **Local tests**:

  ```bash
  pip install -r backend/requirements.txt -r frontend/requirements.txt
  pip install pytest pytest-cov mypy ruff
  pytest -q
  ```

Tests cover forecasts/procure, batch selection, backtesting, catalog IDs, configs, approvals/audit log, data validation, and auth/rate-limit behavior.

---

## Docker / Compose notes

* **Root compose** (`docker-compose.yml`) reads **`./.env`** automatically.
* **Infra compose** (`infra/docker-compose.yml`) reads **`../.env`** via `env_file`. Run it from `infra/`.

If you leave `GEMINI_API_KEY` empty, `/procure/recommendations/explain` will return **404** (by design).

---

## Render deployment

Two services defined in `infra/render.yaml`:

* **agentive-api** (rootDir `backend`)

  * Build: `pip install -r requirements.txt`
  * Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
  * Env: `GEMINI_API_KEY` (optional), `DATA_DIR=/data`, `CONFIG_DIR=/app/configs`, `API_TOKEN` (optional), `CORS_ORIGINS`
  * Disk: attach a persistent disk (2–5 GB) mounted at `/data` for **audit logs** and **model cache**
* **agentive-ui** (rootDir `frontend`)

  * Build: `pip install -r requirements.txt`
  * Start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
  * Env: `API_URL=https://<your-api>.onrender.com/api/v1`

---

## API smoke tests (PowerShell)

```powershell
# First row id from the sales file
Get-Content .\data\sales_train_validation.csv -TotalCount 2 | Select-Object -Last 1

# Replace <SKU_ID> and <TOKEN> if auth is enabled
$base = "http://localhost:8000/api/v1"
$hdr  = @{ Authorization = "Bearer <TOKEN>" }

iwr "$base/catalog/ids?limit=5" -Headers $hdr
iwr "$base/forecasts/<SKU_ID>?horizon_days=28" -Headers $hdr

$body = @{ sku_id = "<SKU_ID>"; horizon_days = 28 } | ConvertTo-Json
iwr -Method Post "$base/procure/recommendations" -Body $body -ContentType "application/json" -Headers $hdr
```

Expected:

* `/forecasts` → `horizon_days` rows with `date, mean, lo, hi, model, confidence`
* `/procure/recommendations` → `order_qty, reorder_point, gmroi_delta, confidence, requires_approval`

---

## Runbook

* **Backtesting**: Use **Backtest**; compare `auto|sma|prophet|xgb`. Toggle history overlay for context. API returns MAPE & PI coverage + recent history.
* **Approvals**: In **Recommendations**, Approve/Reject with a note. Review **Audit Log** for the timeline. Attach a Render disk to persist across deploys.
* **Tuning**:

  * Service level, lead times, carrying cost → `configs/settings.yaml`
  * Guardrails (`auto_approval_limit`, `min_service_level`, `gmroi_min`) → `configs/thresholds.yaml`

---

## Notes & roadmap

Implemented:

* Typeahead & caching, CSV exports, health banner
* Gemini explanations (optional; 404 when disabled, optional fallback)
* Config edit & persistence, approvals + audit log
* Rolling backtesting + per-origin coverage + history overlay
* GMROI proxy w/ seasonality & store-level prices
* Model caching under `/data/models`
* n8n daily and approval workflows
* CI + Render + Compose, Prometheus `/metrics`
* Observability: richer logs (now include `sku_id` for POSTs)

Next:

* Deeper backtesting (category/store cross-validation)
* Richer dashboard KPIs and per-store views
* Advanced GMROI (event calendars, price mix over time)
* Stronger auth (JWT/OIDC) and multi-tenant hardening
