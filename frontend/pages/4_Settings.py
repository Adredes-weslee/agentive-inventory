"""
Settings page for configuring business parameters.

This page loads the YAML configuration files under ``configs/`` and displays
their contents.  In a future version you could allow editing and saving
these values directly from the UI.
"""

import os
import yaml
import streamlit as st

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "settings.yaml")
THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "thresholds.yaml")

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