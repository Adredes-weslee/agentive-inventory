"""
Recommendations page for procurement planning.

This page allows users to request a reorder recommendation for a SKU
and display the suggested reorder point and quantity.  It also
generates a simple explanation of the recommendation.
"""

import os
import requests
import streamlit as st
import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.title("📦 Recommendations")

st.write(
    "Generate purchasing recommendations based on demand forecasts and guardrails. "
    "Use this page to decide whether to place an order, and to understand the reasoning behind the suggestion."
)

with st.form(key="recommend_form"):
    sku = st.text_input("SKU ID", value="HOBBIES_1_001")
    horizon = st.number_input("Horizon (days)", min_value=1, max_value=90, value=28, step=1)
    submitted = st.form_submit_button("Get recommendation")

if submitted:
    try:
        with st.spinner("Computing recommendation…"):
            resp = requests.post(
                f"{API_URL}/procure/recommendations",
                json={"sku_id": sku, "horizon_days": horizon},
                timeout=30,
            )
            resp.raise_for_status()
            recs = resp.json()
        if not recs:
            st.info("No recommendation available.")
        else:
            df = pd.DataFrame(recs)
            st.write("### Recommendation")
            st.table(df)
            # Generate a simple explanation without calling the LLM
            rec = recs[0]
            explanation = (
                f"For SKU **{rec['sku_id']}**, the system recommends reordering **{rec['order_qty']}** units "
                f"once the on‑hand inventory falls to **{rec['reorder_point']}** units. "
                f"This suggestion aims to maintain service levels while respecting cash constraints."
            )
            if rec.get("requires_approval"):
                explanation += " Because the order quantity exceeds the auto‑approval limit, a human must approve this purchase."
            st.markdown("### Explanation")
            st.write(explanation)
    except Exception as e:
        st.error(f"Failed to compute recommendation: {e}")