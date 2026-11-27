"""
Train, evaluate, save, load, and predict for the HAR classifier.

The saved artifact is a dict ("bundle") so Infer can check window/feature metadata.
"""

from __future__ import annotations

import io
from typing import Any, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from har.features import (
    OVERLAP,
    WINDOW_SEC,
    build_feature_matrix,
    feature_names,
)

ModelType = Literal["random_forest", "logistic_regression"]


def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: ModelType = "random_forest",
    random_state: int = 42,
) -> Any:
    """Fit a simple classical classifier. Logistic Regression uses scaling."""
    if model_type == "random_forest":
        clf: Any = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight="balanced",
        )
        clf.fit(X, y)
        return clf

    if model_type == "logistic_regression":
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        pipe.fit(X, y)
        return pipe

    raise ValueError(f"Unknown model_type: {model_type}")


def evaluate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Hold out a test split, retrain on train split, and return metrics.

    For tiny classroom datasets this is imperfect but keeps the demo simple.
    """
    labels = sorted(y.unique().tolist())

    # Stratify only when every class has at least 2 samples.
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 and len(labels) > 1 else None

    if len(X) < 4 or stratify is None and len(X) < 4:
        # Too little data for a split — evaluate on training data with a warning flag.
        y_pred = model.predict(X)
        report = classification_report(
            y, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "confusion_matrix": confusion_matrix(y, y_pred, labels=labels),
            "labels": labels,
            "classification_report": report,
            "train_size": int(len(X)),
            "test_size": int(len(X)),
            "in_sample_eval": True,
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Re-fit a fresh model of the same type on the train split for honest metrics.
    if isinstance(model, Pipeline):
        eval_model = train_classifier(X_train, y_train, model_type="logistic_regression")
    else:
        eval_model = train_classifier(X_train, y_train, model_type="random_forest")

    y_pred = eval_model.predict(X_test)
    report = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "labels": labels,
        "classification_report": report,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "in_sample_eval": False,
        "eval_model": eval_model,
    }


def make_bundle(
    model: Any,
    *,
    model_type: ModelType,
    class_labels: list[str],
    sample_rate_hz: float,
    window_sec: float = WINDOW_SEC,
    overlap: float = OVERLAP,
) -> dict[str, Any]:
    """Package model + metadata for joblib export."""
    return {
        "model": model,
        "model_type": model_type,
        "feature_names": feature_names(),
        "class_labels": list(class_labels),
        "window_sec": float(window_sec),
        "overlap": float(overlap),
        "sample_rate_hz": float(sample_rate_hz),
    }


def save_bundle(bundle: dict[str, Any], path_or_buffer: Any) -> None:
    joblib.dump(bundle, path_or_buffer)


def bundle_to_bytes(bundle: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer)
    return buffer.getvalue()


def load_bundle(path_or_buffer: Any) -> dict[str, Any]:
    bundle = joblib.load(path_or_buffer)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Invalid model file. Expected a HAR joblib bundle.")
    for key in ("feature_names", "window_sec", "overlap"):
        if key not in bundle:
            raise ValueError(f"Model bundle missing metadata key: {key}")
    return bundle


def predict_windows(
    bundle: dict[str, Any],
    sensor_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the same windowing/features as training and return per-window predictions.
    """
    window_sec = float(bundle["window_sec"])
    overlap = float(bundle["overlap"])
    # Prefer the recording's own rate; fall back to training rate if estimation fails.
    X, _, hz = build_feature_matrix(
        sensor_df,
        window_sec=window_sec,
        overlap=overlap,
        sample_rate_hz=None,
        require_label=False,
    )

    expected = list(bundle["feature_names"])
    if list(X.columns) != expected:
        X = X.reindex(columns=expected)

    model = bundle["model"]
    preds = model.predict(X)

    result = pd.DataFrame({"predicted_label": preds})
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        conf = proba.max(axis=1)
        result["confidence"] = conf
        for i, cls in enumerate(classes):
            result[f"proba_{cls}"] = proba[:, i]
    else:
        result["confidence"] = np.nan

    result.attrs["sample_rate_hz"] = hz
    return result


def majority_prediction(pred_df: pd.DataFrame) -> tuple[str, float]:
    """Return (majority label, mean confidence of that label's windows)."""
    if pred_df.empty:
        raise ValueError("No windows to summarize.")
    label = str(pred_df["predicted_label"].mode().iloc[0])
    subset = pred_df.loc[pred_df["predicted_label"] == label, "confidence"]
    conf = float(subset.mean()) if len(subset) else float("nan")
    return label, conf
