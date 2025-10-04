"""
Streamlit multipage application for the Agentive Inventory system.

This file configures global options and provides a simple welcome page.
Individual pages live in the ``pages/`` subdirectory; Streamlit will
automatically load them.  To run the app locally use:

```bash
streamlit run app.py
```
"""

import streamlit as st

st.set_page_config(page_title="Agentive Inventory", layout="wide")

st.title("Agentive Inventory Management")

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