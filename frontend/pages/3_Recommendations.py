r"""frontend/pages/3_Recommendations.py

Streamlit page for procurement recommendations."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

from utils.api import get_api_token, get_headers

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")
CATALOG_LIMIT = 500  # backend validation caps limit at 500


def _fetch_recommendations(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    response = requests.post(
        f"{API_URL}/procure/recommendations",
        json=payload,
        headers=get_headers(),
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
            headers=get_headers(),
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


def _render_recommendation_view(body: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> None:
    """Render the recommendation table, rationale, and approval form.

    Persists the request/response in session_state so reruns triggered by
    widget interaction continue displaying the latest result.
    """

    # Persist the latest result so subsequent reruns still show it
    st.session_state["rec_view"] = {"request": body, "recommendations": recommendations}

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

    # ---- Approval UI (wrapped in a form so toggles/typing don't wipe the page) ----
    st.markdown("### Approve or reject recommendation")

    # Extract default recommended quantity from the first row if present
    recommended_qty: Optional[int] = None
    if not df.empty and "order_qty" in df.columns:
        raw_qty = df.iloc[0]["order_qty"]
        try:
            recommended_qty = int(raw_qty)
        except (TypeError, ValueError):
            try:
                recommended_qty = int(float(raw_qty))
            except (TypeError, ValueError):
                recommended_qty = None

    # One decision tracked per-SKU
    approval_state_key = f"approval_state::{body['sku_id']}"
    approval_state = st.session_state.setdefault(approval_state_key, {"completed_action": None})
    interaction_disabled = approval_state.get("completed_action") is not None

    with st.form(f"approval_form::{body['sku_id']}", clear_on_submit=False):
        with st.expander("Approval workflow", expanded=requires_approval and not interaction_disabled):
            provide_override = st.checkbox(
                "Provide quantity override",
                value=False if recommended_qty is not None else True,
                disabled=interaction_disabled,
                key=f"approval_qty_override::{body['sku_id']}",
            )
            if provide_override:
                qty = st.number_input(
                    "Override qty",
                    min_value=0,
                    value=(recommended_qty if (recommended_qty is not None and recommended_qty >= 0) else 0),
                    step=1,
                    disabled=interaction_disabled,
                    key=f"approval_qty_input::{body['sku_id']}",
                )
            else:
                qty = max(int(recommended_qty or 0), 0)

            note = st.text_input(
                "Reason / note",
                value="",
                disabled=interaction_disabled,
                key=f"approval_reason::{body['sku_id']}",
            )

            col_a, col_b = st.columns(2)
            approve = col_a.form_submit_button("✅ Approve", disabled=interaction_disabled)
            reject = col_b.form_submit_button("❌ Reject", disabled=interaction_disabled)

        if approve or reject:
            action = "approve" if approve else "reject"
            submission = {
                "sku_id": body["sku_id"],
                "action": action,
                # Always send qty (API requires ge=0). Use override or recommended.
                "qty": int(qty if isinstance(qty, (int, float)) else 0),
                "reason": (note or action),
            }
            try:
                resp = requests.post(
                    f"{API_URL}/approvals",
                    json=submission,
                    headers=get_headers(),
                    timeout=20,
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                st.error(f"Failed to submit {action} decision: {exc}")
            else:
                approval_state["completed_action"] = action
                st.session_state[approval_state_key] = approval_state
                icon = "✅" if action == "approve" else "❌"
                st.success(f"{icon} {action.title()}d recommendation for {body['sku_id']}")


@st.cache_data(ttl=300)
def _get_catalog_ids(limit: int = 50, api_token: str = "") -> List[str]:
    """Fetch a sample of catalog row identifiers for convenience selections."""

    try:
        response = requests.get(
            f"{API_URL}/catalog/ids",
            params={"limit": limit},
            headers=get_headers(api_token),
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json() or {}
        ids = payload.get("ids") or []
        return [str(sku_id) for sku_id in ids if sku_id]
    except Exception:
        return []


st.title("📦 Recommendations")
st.caption("Compute EOQ/ROP-based reorder guidance backed by forecasting guardrails.")

tab_single, tab_batch = st.tabs(["Single SKU", "Batch"])

with tab_single:
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
                _render_recommendation_view(body, recommendations)
    else:
        # No new submit; if we have a previous result, keep showing it
        rec_view = st.session_state.get("rec_view")
        if rec_view and isinstance(rec_view, dict):
            body = rec_view.get("request") or {}
            recs = rec_view.get("recommendations") or []
            if body and recs:
                _render_recommendation_view(body, recs)


with tab_batch:
    st.subheader("Batch recommendations")
    st.caption(
        "Start typing to select SKUs from the catalog (up to 500) or paste a list manually. "
        "Recommendations will be prioritised by GMROI delta when applying a budget."
    )

    catalog_options = _get_catalog_ids(limit=CATALOG_LIMIT, api_token=get_api_token())
    selected_catalog_ids: List[str] = []

    with st.form(key="batch_form"):
        if catalog_options:
            selected_catalog_ids = st.multiselect(
                "Choose SKUs from catalog (optional)",
                options=catalog_options,
                placeholder="Search SKUs…",
                help="Select one or more IDs fetched from the catalog service.",
            )
        else:
            st.info(
                "Catalog IDs could not be fetched right now. You can still paste SKUs manually."
            )

        sku_block = st.text_area(
            "SKU ids (one per line)",
            value="FOODS_3_090_CA_1_validation\nHOBBIES_1_002_CA_1_validation",
            height=120,
            help="Paste a newline-separated list of M5 row identifiers.",
        )
        batch_horizon = st.slider(
            "Horizon (days)",
            min_value=7,
            max_value=90,
            value=28,
            step=1,
            help="Number of days to use when computing recommendations.",
        )
        cash_budget_value = st.number_input(
            "Cash budget (optional)",
            min_value=0.0,
            value=0.0,
            step=100.0,
            help="If provided, only recommendations within this spend will be auto-selected.",
        )
        run_batch = st.form_submit_button("Run batch")

    if run_batch:
        manual_ids = [sku.strip() for sku in sku_block.splitlines() if sku.strip()]
        combined_ids: List[str] = []
        for sku_id in manual_ids:
            if sku_id not in combined_ids:
                combined_ids.append(sku_id)
        for sku_id in selected_catalog_ids:
            if sku_id not in combined_ids:
                combined_ids.append(sku_id)

        if not combined_ids:
            st.warning("Provide at least one SKU id to request batch recommendations.")
        else:
            payload: Dict[str, Any] = {
                "sku_ids": combined_ids,
                "horizon_days": int(batch_horizon),
            }
            if cash_budget_value and cash_budget_value > 0:
                payload["cash_budget"] = float(cash_budget_value)

            try:
                with st.spinner("Requesting batch recommendations from API…"):
                    response = requests.post(
                        f"{API_URL}/procure/batch_recommendations",
                        json=payload,
                        headers=get_headers(),
                        timeout=60,
                    )
                    response.raise_for_status()
                    batch_data = response.json()
            except requests.Timeout:
                st.error("The batch recommendation request timed out. Please retry shortly.")
            except requests.HTTPError as exc:
                detail = exc.response.json().get("detail") if exc.response is not None else str(exc)
                st.error(f"Batch request failed: {detail}")
            except Exception as exc:  # pragma: no cover - UI fallback
                st.error(f"Unable to fetch batch recommendations: {exc}")
            else:
                total_selected = float(batch_data.get("total_spend_selected", 0.0) or 0.0)
                st.metric("Total spend (selected)", f"${total_selected:,.2f}")

                batch_recommendations = pd.DataFrame(batch_data.get("recommendations", []))
                if batch_recommendations.empty:
                    st.info("No recommendations were returned for the requested SKUs.")
                else:
                    ordered_columns = [
                        "selected",
                        "sku_id",
                        "order_qty",
                        "total_spend",
                        "gmroi_delta",
                        "requires_approval",
                    ]
                    display_columns = [
                        column for column in ordered_columns if column in batch_recommendations.columns
                    ]
                    if display_columns:
                        batch_recommendations = batch_recommendations[display_columns]
                    if "selected" in batch_recommendations.columns:
                        batch_recommendations = batch_recommendations.sort_values(
                            by=["selected"], ascending=False, kind="stable"
                        )
                    st.dataframe(batch_recommendations, use_container_width=True)
