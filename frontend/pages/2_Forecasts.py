"""frontend/pages/2_Forecasts.py

Streamlit page for per-SKU demand forecasts."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

from utils.api import get_api_token, get_headers

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")


@st.cache_data(ttl=300)
def _get_catalog_ids(limit: int = 20, api_token: str = "") -> List[str]:
    try:
        response = requests.get(
            f"{API_URL}/catalog/ids",
            params={"limit": limit},
            headers=get_headers(api_token),
            timeout=15,
        )
        response.raise_for_status()
        return response.json().get("ids", [])
    except Exception:
        return []


@st.cache_data(ttl=300)
def _fetch_forecast(sku_id: str, horizon_days: int, api_token: str = "") -> pd.DataFrame:
    """Fetch forecast rows for a SKU and normalise expected columns."""

    response = requests.get(
        f"{API_URL}/forecasts/{sku_id}",
        params={"horizon_days": int(horizon_days)},
        headers=get_headers(api_token),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json() or {}
    rows: List[Dict[str, Any]] = payload.get("forecast") or []
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    rename_map = {"mean": "mean_units", "lo": "lower", "hi": "upper"}
    df = df.rename(columns=rename_map)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    ordered = [
        col
        for col in ["date", "mean_units", "lower", "upper", "confidence", "model"]
        if col in df.columns
    ]
    return df[ordered] if ordered else df


def _persist_state(key: str, df: Optional[pd.DataFrame], sku_id: str, horizon_days: int) -> None:
    """Store the latest forecast output + CSV bytes in session state."""

    state = st.session_state.setdefault(key, {})
    if df is None or df.empty:
        state.clear()
        st.session_state[key] = state
        return

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    filename = f"forecast_{sku_id}_{horizon_days}d.csv"
    state.update(
        {
            "df": df,
            "csv": csv_bytes,
            "filename": filename,
            "sku_id": sku_id,
            "horizon_days": horizon_days,
        }
    )
    st.session_state[key] = state


def _get_state(key: str) -> Dict[str, Any]:
    return st.session_state.get(key, {})


st.title("🔮 Forecasts")
st.caption("Visualise deterministic forecasts generated from the Walmart M5 dataset.")

STATE_KEY = "forecasts_state"
state = _get_state(STATE_KEY)

examples = _get_catalog_ids(limit=20, api_token=get_api_token())
with st.form(key="forecast_form", clear_on_submit=False):
    if examples:
        default_sku = state.get("sku_id") or examples[0]
        if default_sku not in examples:
            options = [default_sku, *[opt for opt in examples if opt != default_sku]]
        else:
            options = examples
        sku_id = st.selectbox(
            "M5 row id (e.g., FOODS_3_090_CA_1_validation)",
            options=options,
            index=options.index(default_sku),
            placeholder="Type/choose an M5 id (not item_id)",
            help="Enter the M5 *row id* exactly as in the CSV id column.",
        )
    else:
        sku_id = st.text_input(
            "M5 row id",
            value=state.get("sku_id", "FOODS_3_090_CA_1_validation"),
            help="Enter the M5 *row id*, not item_id. Example: FOODS_3_090_CA_1_validation",
        )

    horizon_days = st.slider(
        "Horizon (days)",
        min_value=7,
        max_value=90,
        value=int(state.get("horizon_days", 28)),
        step=1,
        help="Number of days to forecast ahead.",
    )
    submitted = st.form_submit_button("Get forecast")

if submitted:
    api_token = get_api_token()
    try:
        with st.spinner("Fetching forecast from API…"):
            df = _fetch_forecast(sku_id.strip(), int(horizon_days), api_token=api_token)
    except requests.Timeout:
        st.error("The forecast request timed out. Please try again or adjust the horizon.")
        _persist_state(STATE_KEY, None, sku_id, int(horizon_days))
    except requests.HTTPError as exc:
        detail = exc.response.json().get("detail") if exc.response is not None else str(exc)
        st.error(f"API error: {detail}")
        _persist_state(STATE_KEY, None, sku_id, int(horizon_days))
    except Exception as exc:  # pragma: no cover - UI fallback
        st.error(f"Failed to retrieve forecast: {exc}")
        _persist_state(STATE_KEY, None, sku_id, int(horizon_days))
    else:
        if df.empty:
            st.warning("Forecast response was empty. Please verify the SKU ID and horizon.")
            _persist_state(STATE_KEY, None, sku_id, int(horizon_days))
        else:
            _persist_state(STATE_KEY, df, sku_id, int(horizon_days))


state = _get_state(STATE_KEY)
df: Optional[pd.DataFrame] = state.get("df")

if df is not None and not df.empty:
    st.write("### Forecast overview")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        label="Download CSV",
        data=state.get("csv", b""),
        file_name=state.get("filename", "forecast.csv"),
        mime="text/csv",
        disabled=not bool(state.get("csv")),
        key=f"dl::{state.get('sku_id', '')}_{state.get('horizon_days', '')}",
    )

    try:
        import altair as alt

        chart_df = df.rename(columns={"mean_units": "mean"}).copy()
        y_column: Optional[str] = None
        if "mean" in chart_df.columns:
            y_column = "mean"
        elif len(chart_df.columns) > 1:
            y_column = chart_df.columns[1]

        if "date" in chart_df.columns and y_column:
            base = alt.Chart(chart_df).encode(x="date:T")
            band = base.mark_area(opacity=0.2, color="steelblue").encode(y="lower:Q", y2="upper:Q")
            line = base.mark_line(color="#1f77b4").encode(y=f"{y_column}:Q")
            chart = (
                band + line
            ).properties(
                width="container",
                height=320,
                title=f"Forecast horizon for {state.get('sku_id')} ({state.get('horizon_days')} days)",
            )
            st.altair_chart(chart, use_container_width=True)
    except Exception:
        pass

    st.info(
        "Forecasts are generated using a smoothed moving-average baseline with calibrated prediction intervals."
    )
else:
    st.info("Enter a SKU and click **Get forecast** to view results.")
