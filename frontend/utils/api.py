import os

import streamlit as st


def get_headers() -> dict:
    """Return default headers for API requests.

    If an API token is present in Streamlit's session state or the
    ``API_TOKEN`` environment variable, include it as a bearer token in the
    ``Authorization`` header.
    """

    token = (st.session_state.get("api_token") or os.getenv("API_TOKEN", "")).strip()
    return {"Authorization": f"Bearer {token}"} if token else {}
