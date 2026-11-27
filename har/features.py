"""
Windowing and feature extraction for phone accelerometer HAR.

Train and Infer both import from this module so the feature logic stays identical.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Pipeline constants — keep in sync with the saved model bundle metadata.
WINDOW_SEC = 2.0
OVERLAP = 0.5
DEFAULT_SAMPLE_RATE_HZ = 50.0
REQUIRED_COLUMNS = ("timestamp", "ax", "ay", "az")


def estimate_sample_rate_hz(df: pd.DataFrame, default: float = DEFAULT_SAMPLE_RATE_HZ) -> float:
    """Estimate Hz from timestamp differences; fall back to default if needed."""
    if "timestamp" not in df.columns or len(df) < 3:
        return default

    ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(ts) < 3:
        return default

    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return default

    median_dt = float(np.median(diffs))

    # Timestamps may be ms or seconds.
    if median_dt > 5:  # likely milliseconds between samples
        median_dt = median_dt / 1000.0

    if median_dt <= 0:
        return default

    hz = 1.0 / median_dt
    if hz < 5 or hz > 500:
        return default
    return float(hz)


def normalize_sensor_df(df: pd.DataFrame, label: Optional[str] = None) -> pd.DataFrame:
    """Standardize column names and optionally attach a label column."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    rename_map = {
        "time": "timestamp",
        "t": "timestamp",
        "acc_x": "ax",
        "acc_y": "ay",
        "acc_z": "az",
        "x": "ax",
        "y": "ay",
        "z": "az",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    out = out[list(REQUIRED_COLUMNS) + (["label"] if "label" in out.columns else [])].copy()
    for col in REQUIRED_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)

    if label is not None:
        out["label"] = label
    elif "label" in out.columns:
        out["label"] = out["label"].astype(str).str.strip().str.lower()

    return out


def window_indices(
    n_samples: int,
    sample_rate_hz: float,
    window_sec: float = WINDOW_SEC,
    overlap: float = OVERLAP,
) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for sliding windows."""
    window_size = max(int(round(window_sec * sample_rate_hz)), 2)
    step = max(int(round(window_size * (1.0 - overlap))), 1)

    if n_samples < window_size:
        return []

    pairs: list[tuple[int, int]] = []
    start = 0
    while start + window_size <= n_samples:
        pairs.append((start, start + window_size))
        start += step
    return pairs


def extract_features(window_df: pd.DataFrame) -> dict[str, float]:
    """Summary features for one window: mean/std/min/max per axis + SMA."""
    ax = window_df["ax"].to_numpy(dtype=float)
    ay = window_df["ay"].to_numpy(dtype=float)
    az = window_df["az"].to_numpy(dtype=float)

    feats: dict[str, float] = {}
    for name, arr in (("ax", ax), ("ay", ay), ("az", az)):
        feats[f"{name}_mean"] = float(np.mean(arr))
        feats[f"{name}_std"] = float(np.std(arr))
        feats[f"{name}_min"] = float(np.min(arr))
        feats[f"{name}_max"] = float(np.max(arr))

    # Signal Magnitude Area — common HAR summary of overall motion energy.
    feats["sma"] = float(np.mean(np.abs(ax) + np.abs(ay) + np.abs(az)))
    return feats


def feature_names() -> list[str]:
    names: list[str] = []
    for axis in ("ax", "ay", "az"):
        names.extend([f"{axis}_mean", f"{axis}_std", f"{axis}_min", f"{axis}_max"])
    names.append("sma")
    return names


def build_feature_matrix(
    df: pd.DataFrame,
    window_sec: float = WINDOW_SEC,
    overlap: float = OVERLAP,
    sample_rate_hz: Optional[float] = None,
    require_label: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.Series], float]:
    """
    Convert a continuous accelerometer table into a window-level feature matrix.

    Returns
    -------
    X : DataFrame of features (one row per window)
    y : Series of labels if present / required, else None
    sample_rate_hz : Hz used for windowing
    """
    clean = normalize_sensor_df(df)
    hz = sample_rate_hz if sample_rate_hz is not None else estimate_sample_rate_hz(clean)
    pairs = window_indices(len(clean), hz, window_sec=window_sec, overlap=overlap)

    if not pairs:
        raise ValueError(
            f"Not enough samples for a {window_sec}s window at ~{hz:.1f} Hz "
            f"(need more recording time)."
        )

    rows: list[dict[str, float]] = []
    labels: list[str] = []

    for start, end in pairs:
        window = clean.iloc[start:end]
        feats = extract_features(window)
        rows.append(feats)

        if "label" in clean.columns:
            # Majority label inside the window (handles slight boundary mixing).
            mode = window["label"].mode()
            labels.append(str(mode.iloc[0]) if len(mode) else str(window["label"].iloc[0]))

    X = pd.DataFrame(rows, columns=feature_names())

    if require_label:
        if not labels:
            raise ValueError("Labeled data is required for training (missing 'label' column).")
        y = pd.Series(labels, name="label")
        return X, y, hz

    y = pd.Series(labels, name="label") if labels else None
    return X, y, hz
