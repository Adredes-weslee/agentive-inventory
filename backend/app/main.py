"""
Main entrypoint for the FastAPI application.

The API exposes endpoints to fetch demand forecasts for a given SKU and to
generate procurement recommendations.  A health endpoint is also provided
for readiness/liveness checks.  Configuration is read from environment
variables and YAML files in `configs/`.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.v1 import approvals, configs, forecasts, procure, health

app = FastAPI(title="Agentive Inventory API", version="0.1.0")

# Allow cross‑origin requests from the Streamlit UI (and others).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production specify your UI domain(s)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include versioned routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(forecasts.router, prefix="/api/v1")
app.include_router(procure.router, prefix="/api/v1")
app.include_router(configs.router, prefix="/api/v1")
app.include_router(approvals.router, prefix="/api/v1")