"""Page 2 — Window, extract features, train, and export a model."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har.features import (
    OVERLAP,
    WINDOW_SEC,
    build_feature_matrix,
    normalize_sensor_df,
)
from har.model import (
    bundle_to_bytes,
    evaluate,
    make_bundle,
    train_classifier,
)

st.set_page_config(page_title="Train | Phone HAR", layout="centered")
st.title("2. Train")
st.write(
    """
Upload one or more labeled CSVs (`timestamp, ax, ay, az, label`), or use the
bundled sample files. The app windows the stream, extracts summary features,
trains a classical classifier, and offers a `.joblib` download.
"""
)

sample_dir = ROOT / "sample_data"
use_samples = st.checkbox("Use bundled sample_data CSVs", value=True)

uploads = st.file_uploader(
    "Upload labeled CSV files",
    type=["csv"],
    accept_multiple_files=True,
)

frames: list[pd.DataFrame] = []

if use_samples:
    for path in sorted(sample_dir.glob("*_sample.csv")):
        frames.append(normalize_sensor_df(pd.read_csv(path)))

if uploads:
    for f in uploads:
        frames.append(normalize_sensor_df(pd.read_csv(f)))

if not frames:
    st.warning("Provide at least one labeled CSV (or enable sample data).")
    st.stop()

data = pd.concat(frames, ignore_index=True)
st.write("### Dataset overview")
st.write(f"Total rows: **{len(data)}**")
st.write(data["label"].value_counts().rename("count").to_frame())

model_choice = st.selectbox(
    "Classifier",
    options=["random_forest", "logistic_regression"],
    format_func=lambda x: {
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
    }[x],
)

st.caption(
    f"Window = {WINDOW_SEC}s, overlap = {int(OVERLAP * 100)}%. "
    "Feature set: mean/std/min/max per axis + SMA."
)

if st.button("Train model", type="primary"):
    try:
        X, y, hz = build_feature_matrix(data)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.write(f"Windows: **{len(X)}** · estimated sample rate: **{hz:.1f} Hz**")
    st.write("Feature preview:")
    st.dataframe(X.head(), use_container_width=True)

    # Metrics use an internal hold-out re-fit; the downloadable model is fit on all windows.
    model = train_classifier(X, y, model_type=model_choice)
    metrics = evaluate(model, X, y)
    final_model = train_classifier(X, y, model_type=model_choice)

    st.write("### Evaluation")
    if metrics.get("in_sample_eval"):
        st.warning(
            "Too few windows for a reliable hold-out split; metrics are in-sample."
        )
    else:
        st.caption(
            f"Hold-out split — train windows: {metrics['train_size']}, "
            f"test windows: {metrics['test_size']}"
        )

    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")

    labels = metrics["labels"]
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)
    plt.close(fig)

    report = metrics["classification_report"]
    rows = []
    for cls in labels:
        rows.append(
            {
                "class": cls,
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1-score": report[cls]["f1-score"],
                "support": report[cls]["support"],
            }
        )
    st.write("Per-class precision / recall")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    bundle = make_bundle(
        final_model,
        model_type=model_choice,
        class_labels=labels,
        sample_rate_hz=hz,
        window_sec=WINDOW_SEC,
        overlap=OVERLAP,
    )
    st.session_state["trained_bundle"] = bundle
    st.success("Model trained and ready to download.")

if "trained_bundle" in st.session_state:
    blob = bundle_to_bytes(st.session_state["trained_bundle"])
    st.download_button(
        label="Download model.joblib",
        data=blob,
        file_name="har_model.joblib",
        mime="application/octet-stream",
    )
