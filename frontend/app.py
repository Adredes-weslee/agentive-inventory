"""
Streamlit multipage application for the Agentive Inventory system.

This file configures global options and provides a simple welcome page.
Individual pages live in the ``pages/`` subdirectory; Streamlit will
automatically load them.  To run the app locally use:

```bash
streamlit run app.py
```
"""

import os
from typing import Final

import requests
import streamlit as st

st.set_page_config(page_title="Agentive Inventory", layout="wide")

st.title("Agentive Inventory Management")

API_URL: Final[str] = os.getenv("API_URL", "http://localhost:8000/api/v1")
status = "⚠️ not reachable"

try:
    response = requests.get(f"{API_URL}/health", timeout=5)
except Exception:
    response = None

if response is not None and response.ok:
    status = "✅ healthy"

st.caption(f"Backend API: {status} — {API_URL}  ·  Set `API_URL` if needed.")

st.markdown(
    """
    Welcome to the Agentive Inventory Management System.  Use the navigation
    sidebar to explore forecasts, review procurement recommendations and
    adjust business settings.  This application communicates with the
    FastAPI backend via REST endpoints; ensure the backend server is
    running and that the API URL is correctly configured in your
    environment variables (see `.env`).
    """
)
