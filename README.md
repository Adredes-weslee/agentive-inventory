# Agentive Inventory Management System

This repository implements a multi‑agent inventory management system built around a set of cooperating services and a human in the loop.  It follows the architecture described in the provided solution documents and uses the **M5 Forecasting** dataset (Walmart daily item sales with promotions, prices and events) as the canonical source of demand history.  The goal of this system is to forecast demand for individual SKUs, generate procurement recommendations that maximise gross margin return on investment (GMROI) subject to business guardrails and allow human operators to review, adjust and approve those recommendations via a simple web interface.

## Repository layout

```
agentive‑inventory/
├── README.md                # This file
├── .env.example             # Example environment variables for local development
├── pyproject.toml           # Development tooling configuration (ruff, mypy, pytest)
├── configs/
│   ├── settings.yaml        # Business/operational context and defaults (MCP fields)
│   └── thresholds.yaml      # Guardrails such as approval limits and service levels
├── data/                    # Place the M5 dataset files here
│   ├── sales_train_validation.csv
│   ├── calendar.csv
│   └── sell_prices.csv
├── backend/                 # FastAPI backend exposing forecasting and procurement APIs
│   ├── app/
│   │   ├── main.py          # Application entrypoint
│   │   ├── api/v1/
│   │   │   ├── forecasts.py # GET /forecasts/{sku_id}
│   │   │   ├── procure.py   # POST /procure/recommendations
│   │   │   └── health.py    # GET /health
│   │   ├── core/config.py   # Settings and configuration loading
│   │   ├── models/schemas.py# Pydantic models used in the API
│   │   └── services/
│   │       ├── inventory_service.py   # Inventory state and SKU metadata access
│   │       ├── forecasting_service.py # Demand forecasting from M5 data
│   │       ├── procurement_service.py # Procurement logic and guardrails
│   │       └── llm_service.py         # Optional Gemini API integration
│   ├── Dockerfile            # Container build for backend
│   └── requirements.txt      # Backend dependencies
├── orchestration/
│   ├── n8n_workflows/        # Example n8n workflow JSON for end‑to‑end orchestration
│   │   └── example_workflow.json
│   └── README.md             # Notes about orchestration
├── frontend/                 # Streamlit user interface
│   ├── app.py               # Top level multipage app
│   ├── pages/
│   │   ├── 1_Dashboard.py   # Key performance indicators (KPIs)
│   │   ├── 2_Forecasts.py   # Forecast visualisation and overrides
│   │   ├── 3_Recommendations.py # Purchase recommendation review and approval
│   │   └── 4_Settings.py    # Adjust guardrails and context
│   └── requirements.txt     # Frontend dependencies
├── infra/
│   ├── docker-compose.yml    # Local development stack: API + UI
│   ├── render.yaml           # Example configuration for deployment on Render
│   └── k8s/                  # Placeholder for optional Kubernetes manifests
└── .github/workflows/
    └── ci.yml               # Continuous integration pipeline using GitHub Actions
```

## Using the M5 dataset

The [M5 Forecasting dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy) consists of three CSV files:

| file                          | description                                                             |
|------------------------------|---------------------------------------------------------------------------|
| `sales_train_validation.csv` | Daily unit sales for each product in each store. Columns `d_1`, … represent successive days. |
| `calendar.csv`               | Mapping from `d_*` columns to actual calendar dates and event annotations. |
| `sell_prices.csv`            | Weekly sale prices of products by store.                                   |

For privacy and licensing reasons these files are **not included** in this repository.  To use the system you must download the dataset from Kaggle and place the CSVs inside the `data/` directory as shown above.  The backend services will automatically load the files when available.

## Quick start

1. Clone this repository and change into the directory:

   ```bash
   git clone <your‑repo>.git
   cd agentive‑inventory
   ```

2. Download the three M5 dataset files from Kaggle and place them in the `data/` folder.

   ```text
   data/
   ├── sales_train_validation.csv
   ├── sell_prices.csv
   └── calendar.csv
   ```

   The service validates that each file exists before serving forecasts or procurement guidance.

3. Copy `.env.example` to `.env` and fill in your configuration, especially your `GEMINI_API_KEY` if you plan to use the optional LLM features.

4. Build and run the backend locally using Docker Compose:

   ```bash
   docker compose up --build
   ```

   This will start the FastAPI backend on `http://localhost:8000` and the Streamlit UI on `http://localhost:8501`.

5. Navigate to the Streamlit UI to explore forecasts and procurement recommendations.

6. (Optional) Run the backend unit tests locally to verify the deterministic forecasting and procurement guardrails:

   ```bash
   poetry run pytest backend/tests
   ```

   The tests use lightweight synthetic M5 extracts so they execute quickly without the full dataset.

### Configuration quick reference

- `configs/settings.yaml` — operational context such as `service_level_target`, `lead_time_days`, `carrying_cost_rate`, `order_cost` and `gross_margin_rate`. These values feed directly into the EOQ/ROP calculations.
- `configs/thresholds.yaml` — guardrails including `auto_approval_limit`, `min_service_level`, `gmroi_min` and `max_cash_outlay`. Recommendations that breach these thresholds are flagged for manual approval in the UI.

## Design philosophy

The system is composed of loosely coupled services communicating via HTTP and orchestrated by an external workflow engine such as [n8n](https://n8n.io/).  Core design principles include:

- **Hybrid forecasting**: combine classical time‑series models (e.g. Prophet) and machine learning models (e.g. XGBoost) based on SKU characteristics such as ABC classification and gross margins.  A naive baseline is provided but can be replaced with more sophisticated models.
- **Guardrails and supervised autonomy**: procurement recommendations follow business rules (minimum service level, maximum cash outlay) and are flagged for human approval when confidence is low or spend is high.
- **Human collaboration hub**: a Streamlit application exposes forecasts, KPIs and recommendations to human operators who can override or approve agent decisions.
- **Extensibility**: additional services (e.g. price optimisation, promotion planning) can be plugged in with minimal changes.  Configuration is externalised into YAML files under `configs/`.

We encourage you to explore the code, adapt the forecasting logic to your needs and extend the orchestration workflows to support your business processes.
