"""
Settings page for configuring business parameters.

This page loads the YAML configuration files under ``configs/`` and displays
their contents.  In a future version you could allow editing and saving
these values directly from the UI.
"""

import os
import requests
import yaml
import streamlit as st

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "settings.yaml")
THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "thresholds.yaml")
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.title("⚙️ Settings")

st.write(
    "The values below are loaded from the YAML files in the `configs/` folder. "
    "Adjust them by editing the files directly or by implementing persistence in the backend."
)


def load_yaml(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


settings_data = load_yaml(SETTINGS_PATH)
thresholds_data = load_yaml(THRESHOLDS_PATH)

st.subheader("Business Context (settings.yaml)")
st.json(settings_data)

st.subheader("Guardrails (thresholds.yaml)")
st.json(thresholds_data)

st.markdown("---")
st.subheader("Edit & Save (experimental)")
with st.form("edit_settings"):
    col1, col2, col3 = st.columns(3)
    with col1:
        sl = st.number_input(
            "service_level_target",
            value=float(settings_data.get("service_level_target", 0.95)),
            min_value=0.5,
            max_value=0.999,
            step=0.01,
        )
        cc = st.number_input(
            "carrying_cost_rate (annual)",
            value=float(settings_data.get("carrying_cost_rate", 0.24)),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        )
    with col2:
        lt = st.number_input(
            "lead_time_days",
            value=int(settings_data.get("lead_time_days", 14)),
            min_value=0,
            max_value=90,
            step=1,
        )
    with col3:
        aal = st.number_input(
            "auto_approval_limit",
            value=float(thresholds_data.get("auto_approval_limit", 1000.0)),
            min_value=0.0,
            step=10.0,
        )
        msl = st.number_input(
            "min_service_level",
            value=float(thresholds_data.get("min_service_level", 0.90)),
            min_value=0.5,
            max_value=0.999,
            step=0.01,
        )
        gm = st.number_input(
            "gmroi_min",
            value=float(thresholds_data.get("gmroi_min", 0.05)),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        )
    submitted = st.form_submit_button("Save")
    if submitted:
        try:
            r1 = requests.put(
                f"{API_URL}/configs/settings",
                json={
                    "service_level_target": sl,
                    "carrying_cost_rate": cc,
                    "lead_time_days": lt,
                },
                timeout=20,
            )
            r2 = requests.put(
                f"{API_URL}/configs/thresholds",
                json={
                    "auto_approval_limit": aal,
                    "min_service_level": msl,
                    "gmroi_min": gm,
                },
                timeout=20,
            )
            if r1.ok and r2.ok:
                st.success("Saved to backend.")
            else:
                messages = []
                if not r1.ok:
                    messages.append(
                        f"settings update failed ({r1.status_code})"
                    )
                if not r2.ok:
                    messages.append(
                        f"thresholds update failed ({r2.status_code})"
                    )
                joined = ", ".join(messages) or "backend did not accept updates"
                st.info(
                    f"Backend did not accept updates ({joined}). Edit YAML files directly for now."
                )
        except Exception as exc:
            st.info(
                f"Backend update not available yet: {exc}. Edit YAML files directly."
            )
