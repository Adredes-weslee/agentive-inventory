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
        # Show KPI cards
        avg_demand = float(df["mean"].mean()) if not df.empty else 0.0

        try:
            settings = requests.get(f"{API_URL}/configs/settings", timeout=10).json()
        except Exception:
            settings = {}

        service_level = settings.get("service_level_target", 0.95)
        if isinstance(service_level, str):
            try:
                service_level = float(service_level)
            except ValueError:
                service_level = 0.95
        elif not isinstance(service_level, (int, float)):
            service_level = 0.95

        lead_time_days = settings.get("lead_time_days", 14)
        if isinstance(lead_time_days, str):
            try:
                lead_time_days = float(lead_time_days)
            except ValueError:
                lead_time_days = 14
        if isinstance(lead_time_days, float):
            lead_time_days = int(round(lead_time_days))
        elif not isinstance(lead_time_days, int):
            lead_time_days = 14

        c1, c2, c3 = st.columns(3)
        c1.metric("Average forecast", f"{avg_demand:.2f}/day")
        c2.metric("Service level target", f"{service_level:.3f}")
        c3.metric("Lead time (days)", f"{lead_time_days}")
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
