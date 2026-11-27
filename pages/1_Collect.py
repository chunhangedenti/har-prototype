"""Page 1 — Collect labeled phone accelerometer data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har.sensor_bridge import motion_recorder, samples_to_dataframe

st.set_page_config(page_title="Collect | Phone HAR", layout="centered")
st.title("1. Collect")
st.write(
    """
Record phone accelerometer data for one activity at a time.
Aim for **30–60 seconds per class** (sitting, walking, running).

Open this app on a **phone** via the Streamlit Community Cloud HTTPS URL.
Desktop browsers usually cannot provide motion sensors.
"""
)

label = st.selectbox("Activity label", ["sitting", "walking", "running"])

st.subheader("Sensor recorder")
payload = motion_recorder(key=f"recorder_{label}")

if "collected_dfs" not in st.session_state:
    st.session_state.collected_dfs = {}

if payload and isinstance(payload, dict) and payload.get("samples"):
    df_new = samples_to_dataframe(payload, label=label)
    st.session_state.collected_dfs[label] = df_new
    st.success(f"Received {len(df_new)} samples for **{label}**.")

df = st.session_state.collected_dfs.get(label)

if df is None or df.empty:
    st.info("No samples yet for this label. Use the recorder above, then tap Stop & send.")
else:
    st.metric("Rows captured", len(df))
    duration_ms = float(df["timestamp"].max() - df["timestamp"].min())
    st.caption(f"Approx. duration: {duration_ms / 1000.0:.1f} s (from timestamps)")

    st.dataframe(df.head(20), use_container_width=True)

    chart_df = df[["timestamp", "ax", "ay", "az"]].set_index("timestamp")
    st.line_chart(chart_df)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {label}.csv",
        data=csv_bytes,
        file_name=f"{label}.csv",
        mime="text/csv",
    )

st.divider()
st.write("### Session summary")
if not st.session_state.collected_dfs:
    st.caption("No recordings in this session yet.")
else:
    summary = {
        lbl: len(frame) for lbl, frame in st.session_state.collected_dfs.items()
    }
    st.write(pd.DataFrame({"label": list(summary.keys()), "rows": list(summary.values())}))
