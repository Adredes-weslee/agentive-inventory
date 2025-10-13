"""Streamlit page for procurement recommendations."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")


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
    """Return a rationale string from the backend (if available), else a heuristic summary."""

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
            "The explanation service timed out. Showing a heuristic summary instead."
        )
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            st.info("Explanation service unavailable; showing heuristic summary instead.")
        else:
            detail = exc.response.json().get("detail") if exc.response is not None else str(exc)
            st.warning(
                f"Explanation service returned an error: {detail}. Showing heuristic summary instead."
            )
    except requests.RequestException as exc:  # pragma: no cover - network failure
        st.warning(
            f"Unable to reach the explanation service: {exc}. Showing heuristic summary instead."
        )
    else:
        data = response.json()
        explanation = data.get("explanation") or data.get("message")
        if isinstance(explanation, str) and explanation.strip():
            return explanation.strip()

    # Heuristic fallback summary if backend explanation unavailable
    try:
        primary_rec = recommendations[0]
        order_qty = primary_rec.get("order_qty")
        reorder_point = primary_rec.get("reorder_point")
        confidence = primary_rec.get("confidence")
        requires_approval = primary_rec.get("requires_approval")

        # Format components defensively in case of missing data
        if order_qty is not None:
            action_text = f"order {order_qty} units"
        else:
            action_text = "adjust order quantities"
        rop_text = (
            f"to stay above ROP={reorder_point}"
            if reorder_point is not None
            else "to maintain stock targets"
        )
        approval_text = (
            "approval required"
            if requires_approval
            else "auto-approval"
            if requires_approval is not None
            else "approval status unknown"
        )
        confidence_text = (
            f"confidence={confidence:.2f}"
            if isinstance(confidence, (int, float))
            else "confidence unavailable"
        )

        return (
            "Heuristic rationale: "
            f"Recommend to {action_text} {rop_text}; {approval_text}; {confidence_text}."
        )
    except Exception:
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
