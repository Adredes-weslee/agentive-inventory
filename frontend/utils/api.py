r"""frontend\utils\api.py"""

import os
from typing import Optional

import streamlit as st


def get_api_token() -> str:
    """Return the API token from session state or the environment."""

    return (st.session_state.get("api_token") or os.getenv("API_TOKEN", "")).strip()


def get_headers(token: Optional[str] = None) -> dict:
    """Return default headers for API requests.

    If an API token is present in Streamlit's session state or the
    ``API_TOKEN`` environment variable, include it as a bearer token in the
    ``Authorization`` header.

    Parameters
    ----------
    token:
        Optional explicit token to use. When ``None`` (the default), the token
        is looked up via :func:`get_api_token` so callers can decide whether the
        token should influence cache keys.
    """

    resolved_token = token.strip() if isinstance(token, str) else get_api_token()
    return {"Authorization": f"Bearer {resolved_token}"} if resolved_token else {}
