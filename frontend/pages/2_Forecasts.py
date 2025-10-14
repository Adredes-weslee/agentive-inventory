"""Streamlit page for per-SKU demand forecasts."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")


@st.cache_data(ttl=300)
def _get_catalog_ids(limit: int = 20) -> List[str]:
    try:
        response = requests.get(
            f"{API_URL}/catalog/ids",
            params={"limit": limit},
            timeout=15,
        )
        response.raise_for_status()
        return response.json().get("ids", [])
    except Exception:
        return []


@st.cache_data(ttl=300)
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

examples = _get_catalog_ids(limit=20)
default_sku = "FOODS_3_090_CA_1_validation"
sku_state_key = "forecast_form_sku"
if sku_state_key not in st.session_state:
    st.session_state[sku_state_key] = examples[0] if examples else default_sku

with st.form(key="forecast_form"):
    sku_input = st.text_input(
        "M5 row id",
        key=sku_state_key,
        help="Enter the M5 *row id*, not item_id. Example: FOODS_3_090_CA_1_validation",
    )
    if examples:
        suggestion = st.selectbox(
            "Cached ids (optional)",
            options=["Type your own id", *examples],
            index=0,
            help="Select a cached id or continue typing to use any valid M5 row id.",
        )
        if suggestion != "Type your own id" and st.session_state[sku_state_key] != suggestion:
            st.session_state[sku_state_key] = suggestion
            sku_input = suggestion
    sku = sku_input.strip()
    horizon = st.slider(
        "Horizon (days)",
        min_value=7,
        max_value=90,
        value=28,
        step=1,
        help="Number of days to forecast ahead.",
    )
    submitted = st.form_submit_button("Get forecast")

if submitted:
    try:
        with st.spinner("Fetching forecast from API…"):
            payload = _fetch_forecast(sku, int(horizon))
    except requests.Timeout:
        st.error("The forecast request timed out. Please try again or adjust the horizon.")
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
            base_columns = ["date", "mean", "lo", "hi", "confidence", "model"]
            table_df = forecast_df[[col for col in base_columns if col in forecast_df.columns]].rename(
                columns={"mean": "mean_units", "lo": "lower", "hi": "upper"}
            )
            st.dataframe(table_df, use_container_width=True)
            st.download_button(
                "Download CSV",
                table_df.to_csv(index=False).encode("utf-8"),
                file_name=f"forecast_{payload['sku_id']}.csv",
                mime="text/csv",
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
