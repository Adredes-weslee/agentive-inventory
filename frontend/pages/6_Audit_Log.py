r"""frontend/pages/6_Audit_Log.py

Audit log page for approvals."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st

from utils.api import get_headers
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")


def _fetch_audit_log(limit: int = 200) -> List[Dict[str, Any]]:
    response = requests.get(
        f"{API_URL}/approvals/audit-log",
        params={"limit": limit},
        headers=get_headers(),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json() or {}
    events = payload.get("events", [])
    return events if isinstance(events, list) else []


st.title("🗒️ Approval Audit Log")
st.caption("Review historical approval and rejection activity.")

with st.spinner("Loading audit log…"):
    try:
        audit_events = _fetch_audit_log()
    except requests.Timeout:
        st.error("Audit log request timed out. Please retry shortly.")
        audit_events = []
    except requests.RequestException as exc:
        st.error(f"Failed to fetch audit events: {exc}")
        audit_events = []

if not audit_events:
    st.info("No audit events yet.")
else:
    df = pd.DataFrame(audit_events)
    if "ts" in df.columns:
        local_tz = datetime.now().astimezone().tzinfo
        df["timestamp"] = (
            pd.to_datetime(df["ts"], unit="s", utc=True)
            .dt.tz_convert(local_tz)
            .dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        )
        df = df.drop(columns=["ts"])

    display_columns = [
        "timestamp",
        "sku_id",
        "action",
        "qty",
        "reason",
        "actor",
    ]
    ordered_cols = [col for col in display_columns if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in ordered_cols]
    df = df[ordered_cols + remaining_cols]

    sort_column = "timestamp" if "timestamp" in df.columns else None
    if sort_column:
        df = df.sort_values(by=sort_column, ascending=False)

    st.dataframe(
        df,
        use_container_width=True,
    )
