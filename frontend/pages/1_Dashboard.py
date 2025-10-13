"""
Dashboard page displaying simple KPIs.

This page provides a high‑level overview of inventory performance.  In a
production system you might query a database or analytics warehouse for
metrics such as fill rate, cash conversion cycle or inventory turnover.
Here we demonstrate how to call the backend API and compute basic
statistics from a forecast.
"""

import os
import requests
import streamlit as st
import pandas as pd
import altair as alt

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.title("📊 Dashboard")

st.write(
    "This dashboard provides a quick overview of demand forecasts. "
    "Select a SKU to view its predicted demand over the next 28 days."
)

sku = st.text_input("SKU ID", "HOBBIES_1_001")

if sku:
    try:
        with st.spinner("Fetching forecast…"):
            resp = requests.get(f"{API_URL}/forecasts/{sku}", timeout=30)
            resp.raise_for_status()
            data = resp.json()
        df = pd.DataFrame(data["forecast"])
        df["date"] = pd.to_datetime(df["date"])
        # Show basic statistics
        avg_demand = df["mean"].mean()
        st.metric("Average forecast demand", f"{avg_demand:.2f} units/day")
        # Plot mean + PI band
        base = alt.Chart(df).encode(x="date:T")
        band = base.mark_area(opacity=0.2).encode(y="lo:Q", y2="hi:Q")
        line = base.mark_line().encode(y="mean:Q")
        st.altair_chart(
            (band + line).properties(width=700, height=300, title=f"Forecast for {sku}"),
            use_container_width=True,
        )
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"dashboard_forecast_{sku}.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Failed to retrieve forecast: {e}")
