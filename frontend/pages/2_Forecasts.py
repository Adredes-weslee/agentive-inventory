"""Streamlit page for per-SKU demand forecasts."""

from __future__ import annotations

import os
from typing import Any, Dict

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")


def _fetch_forecast(sku: str, horizon: int) -> Dict[str, Any]:
    response = requests.get(
        f"{API_URL}/forecasts/{sku}",
        params={"horizon_days": horizon},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


st.title("🔮 Forecasts")
st.caption("Visualise deterministic forecasts generated from the Walmart M5 dataset.")

with st.form(key="forecast_form"):
    sku = st.text_input("SKU ID", value="HOBBIES_1_001", help="Enter an M5 item_id value, e.g. FOODS_3_090")
    horizon = st.number_input("Horizon (days)", min_value=7, max_value=90, value=28, step=1)
    submitted = st.form_submit_button("Get forecast")

if submitted:
    try:
        with st.spinner("Fetching forecast from API…"):
            payload = _fetch_forecast(sku, int(horizon))
    except requests.HTTPError as exc:
        detail = exc.response.json().get("detail") if exc.response is not None else str(exc)
        st.error(f"API error: {detail}")
    except Exception as exc:  # pragma: no cover - UI fallback
        st.error(f"Failed to retrieve forecast: {exc}")
    else:
        forecast_df = pd.DataFrame(payload.get("forecast", []))
        if forecast_df.empty:
            st.warning("Forecast response was empty. Please verify the SKU ID and horizon.")
        else:
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])
            st.write("### Forecast overview")
            st.dataframe(
                forecast_df[["date", "mean", "lo", "hi", "confidence", "model"]].rename(
                    columns={"mean": "mean_units", "lo": "lower", "hi": "upper"}
                ),
                use_container_width=True,
            )

            base = alt.Chart(forecast_df).encode(x="date:T")
            band = base.mark_area(opacity=0.2, color="steelblue").encode(y="lo:Q", y2="hi:Q")
            line = base.mark_line(color="#1f77b4").encode(y="mean:Q")
            chart = (band + line).properties(
                width="container",
                height=320,
                title=f"Forecast horizon for {payload['sku_id']} ({payload['horizon_days']} days)",
            )
            st.altair_chart(chart, use_container_width=True)

            st.info(
                "Forecasts are generated using a smoothed moving-average baseline with calibrated prediction intervals."
            )
