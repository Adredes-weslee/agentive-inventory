"""Streamlit page for running historical forecast backtests."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import altair as alt
import pandas as pd
import requests
import streamlit as st

from ..utils.api import get_headers
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.title("🧪 Backtest")

st.write(
    "Run a rolling-origin backtest against the demand forecasts to evaluate "
    "model performance for a given SKU."
)

with st.form("backtest_form"):
    sku = st.text_input("SKU (M5 row id)", value="FOODS_3_090_CA_1_validation")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        window = st.number_input("Window (days)", min_value=7, max_value=365, value=56, step=1)
    with col2:
        horizon = st.number_input("Horizon (days)", min_value=1, max_value=60, value=28, step=1)
    with col3:
        step = st.number_input("Step (days)", min_value=1, max_value=30, value=7, step=1)
    with col4:
        model = st.selectbox("Model", ["auto", "sma", "prophet", "xgb"], index=0)

    submitted = st.form_submit_button("Run backtest")

if submitted:
    with st.spinner("Running backtest..."):
        try:
            response = requests.get(
                f"{API_URL}/backtest/{sku}",
                params={
                    "window": window,
                    "horizon": horizon,
                    "step": step,
                    "model": model,
                },
                headers=get_headers(),
                timeout=90,
            )
        except Exception as exc:  # pragma: no cover - network errors
            st.error(f"Could not reach backend: {exc}")
        else:
            if response.ok:
                payload: Dict[str, Any] = response.json()
                dates: List[str] = payload.get("dates", [])
                y = payload.get("y", [])
                yhat = payload.get("yhat", [])

                if not dates:
                    st.warning("Backend returned no backtest data for this configuration.")
                else:
                    df = pd.DataFrame({
                        "date": pd.to_datetime(dates),
                        "y": y,
                        "yhat": yhat,
                    })

                    history_dates = payload.get("history_dates") or []
                    history_values = payload.get("history_values") or []
                    history_df = None
                    if history_dates and history_values:
                        history_df = pd.DataFrame(
                            {
                                "date": pd.to_datetime(history_dates),
                                "history": history_values,
                            }
                        )

                    cols = st.columns(3)
                    with cols[0]:
                        mape = payload.get("mape")
                        st.metric(
                            "MAPE",
                            "n/a" if mape is None else f"{float(mape) * 100:.2f}%",
                        )
                    with cols[1]:
                        coverage = payload.get("coverage")
                        st.metric(
                            "PI coverage",
                            "n/a" if coverage is None else f"{float(coverage) * 100:.1f}%",
                        )
                    with cols[2]:
                        st.metric("Points", f"{len(df):,}")

                    st.caption(f"Model used: {payload.get('model_used', 'n/a')}")

                    base = alt.Chart(df).encode(x="date:T")
                    layers = [
                        base.mark_line(color="#1f77b4").encode(y="y:Q", tooltip=["date", "y"]),
                        base.mark_line(color="#ff7f0e").encode(
                            y="yhat:Q", tooltip=["date", "yhat"]
                        ),
                    ]

                    if history_df is not None:
                        history_chart = (
                            alt.Chart(history_df)
                            .mark_line(color="#2ca02c", strokeDash=[4, 2], opacity=0.6)
                            .encode(x="date:T", y="history:Q", tooltip=["date", "history"])
                        )
                        layers.append(history_chart)

                    forecast_chart = alt.layer(*layers).interactive()
                    st.altair_chart(
                        forecast_chart.properties(title="Forecast vs. actuals with history overlay"),
                        use_container_width=True,
                    )

                    per_origin = payload.get("per_origin_coverage") or []
                    if per_origin:
                        st.write("### Per-origin coverage")
                        coverage_df = pd.DataFrame(
                            {
                                "Origin": list(range(1, len(per_origin) + 1)),
                                "Coverage": [float(value) for value in per_origin],
                            }
                        )
                        coverage_df["Trend"] = [
                            coverage_df.loc[: index, "Coverage"].tolist()
                            for index in coverage_df.index
                        ]

                        st.dataframe(
                            coverage_df,
                            use_container_width=True,
                            column_config={
                                "Origin": st.column_config.NumberColumn("Origin", format="%d"),
                                "Coverage": st.column_config.ProgressColumn(
                                    "Coverage",
                                    format="{:.0%}",
                                    min_value=0.0,
                                    max_value=1.0,
                                ),
                                "Trend": st.column_config.LineChartColumn(
                                    "Trend",
                                    y_min=0.0,
                                    y_max=1.0,
                                ),
                            },
                            hide_index=True,
                        )

                    csv_data = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download CSV",
                        data=csv_data,
                        file_name=f"backtest_{sku}.csv",
                        mime="text/csv",
                    )
            else:
                try:
                    detail = response.json()
                except ValueError:
                    detail = response.text
                st.error(f"Backtest failed ({response.status_code}): {detail}")
else:
    st.info("Submit the form to run a backtest.")
