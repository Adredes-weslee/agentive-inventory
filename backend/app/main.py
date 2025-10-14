r"""backend\app\main.py

Main entrypoint for the FastAPI application.

The API exposes endpoints to fetch demand forecasts for a given SKU and to
generate procurement recommendations.  A health endpoint is also provided
for readiness/liveness checks.  Configuration is read from environment
variables and YAML files in `configs/`.
"""


from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import os
import logging
from pathlib import Path
from .api.v1 import (
    approvals,
    backtest,
    catalog,
    configs,
    data,
    forecasts,
    health,
    procure,
)
from .core.observability import TokenAndRateLimitMiddleware, metrics_endpoint

# Load .env from repo root
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    BASE_DIR = Path(__file__).resolve().parents[2]  # repo root
    load_dotenv(BASE_DIR / ".env")
except Exception:
    pass


logging.getLogger(__name__).info(
    "LLM enabled: %s model=%s",
    bool(os.getenv("GEMINI_API_KEY")),
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
)

app = FastAPI(title="Agentive Inventory API", version="0.1.0")

# Allow cross-origin requests from the Streamlit UI (and others).
origins_env = os.getenv("CORS_ORIGINS", "")
origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # In production specify your UI domain(s)
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TokenAndRateLimitMiddleware)

# Include versioned routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(forecasts.router, prefix="/api/v1")
app.include_router(procure.router, prefix="/api/v1")
app.include_router(catalog.router, prefix="/api/v1")
app.include_router(configs.router, prefix="/api/v1")
app.include_router(approvals.router, prefix="/api/v1")
app.include_router(backtest.router, prefix="/api/v1")
app.include_router(data.router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def _root() -> RedirectResponse:
    """Redirect the root path to the interactive docs."""

    return RedirectResponse(url="/docs")


@app.get("/metrics", include_in_schema=False)
async def _metrics() -> Response:
    """Expose Prometheus metrics."""

    return metrics_endpoint()
