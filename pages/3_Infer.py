"""Page 3 — Load a saved model and predict activity labels."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har.features import normalize_sensor_df
from har.model import load_bundle, majority_prediction, predict_windows
from har.sensor_bridge import motion_recorder, samples_to_dataframe

st.set_page_config(page_title="Infer | Phone HAR", layout="centered")
st.title("3. Infer")
st.write(
    """
Upload a `.joblib` model from the Train page, then either capture a fresh live
stream or upload a new CSV. Predictions use the **same** windowing and feature
code as training (`har/features.py`).
"""
)

model_file = st.file_uploader("Upload har_model.joblib", type=["joblib", "pkl"])

bundle = None
if model_file is not None:
    try:
        bundle = load_bundle(model_file)
        st.success(
            f"Loaded **{bundle.get('model_type', 'model')}** · "
            f"window={bundle['window_sec']}s · overlap={bundle['overlap']}"
        )
        st.caption(
            "Classes: "
            + ", ".join(bundle.get("class_labels") or [])
            + " · features: "
            + ", ".join(bundle["feature_names"])
        )
    except Exception as exc:  # noqa: BLE001 — show friendly UI error
        st.error(f"Could not load model: {exc}")
        st.stop()
elif "trained_bundle" in st.session_state:
    bundle = st.session_state["trained_bundle"]
    st.info("Using the model trained earlier in this session (Train page).")
else:
    st.warning("Upload a model file, or train one on the Train page first.")
    st.stop()

st.subheader("Input data")
mode = st.radio("Data source", ["Upload CSV", "Live phone capture"], horizontal=True)

sensor_df: pd.DataFrame | None = None

if mode == "Upload CSV":
    csv_file = st.file_uploader("Accelerometer CSV", type=["csv"], key="infer_csv")
    use_sample = st.selectbox(
        "Or pick a bundled sample",
        ["(none)", "sitting_sample.csv", "walking_sample.csv", "running_sample.csv"],
    )
    if csv_file is not None:
        sensor_df = normalize_sensor_df(pd.read_csv(csv_file))
    elif use_sample != "(none)":
        sensor_df = normalize_sensor_df(pd.read_csv(ROOT / "sample_data" / use_sample))
else:
    st.caption("Record on a phone, then tap Stop & send.")
    payload = motion_recorder(key="infer_recorder")
    if payload and isinstance(payload, dict) and payload.get("samples"):
        sensor_df = samples_to_dataframe(payload, label=None)
        st.session_state["infer_live_df"] = sensor_df
    sensor_df = st.session_state.get("infer_live_df")

if sensor_df is None or sensor_df.empty:
    st.info("Provide sensor data to run inference.")
    st.stop()

st.write(f"Loaded **{len(sensor_df)}** rows")
st.dataframe(sensor_df.head(15), use_container_width=True)

if st.button("Run inference", type="primary"):
    try:
        pred_df = predict_windows(bundle, sensor_df)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    label, conf = majority_prediction(pred_df)
    st.metric("Predicted activity", label)
    st.metric("Mean confidence (majority class)", f"{conf:.3f}")

    st.write("### Per-window predictions")
    st.dataframe(pred_df, use_container_width=True)

    # Simple confidence chart across windows
    if "confidence" in pred_df.columns:
        st.line_chart(pred_df[["confidence"]])
