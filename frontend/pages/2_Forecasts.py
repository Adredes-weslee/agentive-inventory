"""
Forecasts page for viewing and adjusting SKU demand forecasts.

Users can select a SKU and forecast horizon, view the predicted demand
and adjust the forecast manually if needed.  Adjustments are for
visualisation only; they are not persisted in the backend.
"""

import os
import requests
import streamlit as st
import pandas as pd
import altair as alt

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.title("🔮 Forecasts")

st.write(
    "Retrieve demand forecasts for a given SKU.  You can adjust the forecast "
    "values to explore different scenarios; these adjustments are not saved "
    "but help you understand potential impacts."
)

with st.form(key="forecast_form"):
    sku = st.text_input("SKU ID", value="HOBBIES_1_001")
    horizon = st.number_input("Horizon (days)", min_value=1, max_value=90, value=28, step=1)
    submitted = st.form_submit_button("Get forecast")

if submitted:
    try:
        with st.spinner("Fetching forecast…"):
            resp = requests.get(f"{API_URL}/forecasts/{sku}", params={"horizon_days": horizon}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        df = pd.DataFrame(data["forecast"])
        df["date"] = pd.to_datetime(df["date"])
        df["adjusted_mean"] = df["mean"]  # start with original values
        st.write("### Forecast table (editable)")
        edited_df = st.data_editor(
            df[["date", "mean", "lo", "hi", "model", "confidence", "adjusted_mean"]],
            num_rows="dynamic",
            use_container_width=True,
            key="data_editor",
        )
        # Plot original and adjusted forecasts
        base = alt.Chart(edited_df).encode(x="date:T")
        line_original = base.mark_line(color="steelblue").encode(y="mean:Q", tooltip=["date:T", "mean:Q"])
        line_adjusted = base.mark_line(color="firebrick").encode(y="adjusted_mean:Q", tooltip=["date:T", "adjusted_mean:Q"])
        band = base.mark_area(opacity=0.2, color="gray").encode(y="lo:Q", y2="hi:Q")
        chart = (band + line_original + line_adjusted).properties(width=700, height=300, title=f"Forecast for {sku}")
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to retrieve forecast: {e}")