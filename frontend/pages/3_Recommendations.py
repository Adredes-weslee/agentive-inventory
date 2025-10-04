"""Streamlit page for procurement recommendations."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.app.services.llm_service import explain_recommendation  # type: ignore  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")


def _fetch_recommendations(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    response = requests.post(
        f"{API_URL}/procure/recommendations",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


st.title("📦 Recommendations")
st.caption("Compute EOQ/ROP-based reorder guidance backed by forecasting guardrails.")

with st.form(key="recommend_form"):
    sku = st.text_input("SKU ID", value="HOBBIES_1_001")
    horizon = st.number_input("Horizon (days)", min_value=7, max_value=90, value=28, step=1)
    lead_time = st.number_input("Assumed lead time (days)", min_value=1, max_value=60, value=7, step=1)
    submitted = st.form_submit_button("Get recommendation")

if submitted:
    body = {
        "sku_id": sku,
        "horizon_days": int(horizon),
        "context": {"lead_time_days": int(lead_time)},
    }
    try:
        with st.spinner("Requesting recommendation from API…"):
            recommendations = _fetch_recommendations(body)
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
            st.write("### Recommended action")
            st.dataframe(df, use_container_width=True)

            requires_approval = any(rec.get("requires_approval") for rec in recommendations)
            approval_status = "⚠️ Manual approval required" if requires_approval else "✅ Auto-approval"
            st.subheader("Decision support")
            st.write(approval_status)

            explanation = explain_recommendation(body["context"], recommendations)
            st.markdown("### Rationale")
            st.write(explanation)
