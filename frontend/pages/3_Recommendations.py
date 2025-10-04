"""Streamlit page for procurement recommendations."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.app.services.llm_service import explain_recommendation  # type: ignore  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def _fetch_recommendations(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    response = requests.post(
        f"{API_URL}/procure/recommendations",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _explain_recommendations(
    sku_id: str,
    horizon_days: int,
    recommendations: List[Dict[str, Any]],
) -> Optional[str]:
    """Return a rationale string using the backend or local LLM service."""

    if not GEMINI_API_KEY:
        return None

    payload = {
        "sku_id": sku_id,
        "horizon_days": horizon_days,
        "recommendations": recommendations,
    }

    try:
        response = requests.post(
            f"{API_URL}/procure/recommendations/explain",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.Timeout:
        st.warning(
            "The explanation service timed out. Showing a locally generated summary instead."
        )
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            # Endpoint not available; fall back to local helper.
            pass
        else:
            detail = exc.response.json().get("detail") if exc.response is not None else str(exc)
            st.warning(f"Explanation service returned an error: {detail}. Using local summary instead.")
    except requests.RequestException as exc:  # pragma: no cover - network failure
        st.warning(f"Unable to reach the explanation service: {exc}. Falling back to local summary.")
    else:
        data = response.json()
        explanation = data.get("explanation") or data.get("message")
        if isinstance(explanation, str) and explanation.strip():
            return explanation.strip()

    try:
        return explain_recommendation({"sku_id": sku_id, "horizon_days": horizon_days}, recommendations)
    except Exception as exc:  # pragma: no cover - defensive fallback
        st.warning(f"Could not generate a local explanation: {exc}")
        return None


st.title("📦 Recommendations")
st.caption("Compute EOQ/ROP-based reorder guidance backed by forecasting guardrails.")

with st.form(key="recommend_form"):
    sku = st.text_input("SKU ID", value="HOBBIES_1_001")
    horizon = st.slider("Horizon (days)", min_value=7, max_value=90, value=28, step=1)
    submitted = st.form_submit_button("Get recommendation")

if submitted:
    body = {"sku_id": sku, "horizon_days": int(horizon)}
    try:
        with st.spinner("Requesting recommendation from API…"):
            recommendations = _fetch_recommendations(body)
    except requests.Timeout:
        st.error("The recommendation request timed out. Please retry in a moment.")
    except requests.HTTPError as exc:
        detail = exc.response.json().get("detail") if exc.response is not None else str(exc)
        st.error(f"API error: {detail}")
    except Exception as exc:  # pragma: no cover - UI fallback
        st.error(f"Failed to compute recommendation: {exc}")
    else:
        if not recommendations:
            st.info("No recommendation available for the requested SKU.")
        else:
            df = pd.DataFrame(recommendations)
            desired_columns = ["order_qty", "reorder_point", "confidence", "requires_approval"]
            display_df = df[[col for col in desired_columns if col in df.columns]]

            st.write("### Recommended action")
            st.dataframe(display_df, use_container_width=True)

            requires_approval = any(rec.get("requires_approval") for rec in recommendations)
            approval_status = "⚠️ Manual approval required" if requires_approval else "✅ Auto-approval"
            st.subheader("Decision support")
            st.write(approval_status)

            explanation = _explain_recommendations(body["sku_id"], body["horizon_days"], recommendations)
            if explanation:
                st.markdown("### Rationale")
                st.write(explanation)
