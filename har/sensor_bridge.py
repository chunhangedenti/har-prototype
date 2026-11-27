"""
Phone accelerometer bridge for Streamlit.

Records DeviceMotion samples entirely in the browser and returns one batch
when the user taps Stop. No sample-by-sample Streamlit reruns.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd
import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

_motion_recorder = components.declare_component(
    "har_motion_recorder",
    path=_FRONTEND_DIR,
)


def motion_recorder(key: str = "motion_recorder") -> Optional[dict[str, Any]]:
    """
    Render the sealed JS recorder and return the latest component value.

    Expected value shape:
      {"samples": [{"timestamp", "ax", "ay", "az"}, ...], "error": None|str}
    """
    return _motion_recorder(key=key, default=None)


def samples_to_dataframe(
    payload: Optional[dict[str, Any]],
    label: Optional[str] = None,
) -> pd.DataFrame:
    """Convert the bridge payload into a dataframe ready for CSV export."""
    if not payload or not isinstance(payload, dict):
        return pd.DataFrame(columns=["timestamp", "ax", "ay", "az", "label"])

    samples = payload.get("samples") or []
    if not samples:
        return pd.DataFrame(columns=["timestamp", "ax", "ay", "az", "label"])

    df = pd.DataFrame(samples)
    for col in ("timestamp", "ax", "ay", "az"):
        if col not in df.columns:
            raise ValueError(f"Sensor payload missing column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "ax", "ay", "az"]).reset_index(drop=True)

    if label is not None:
        df["label"] = label
    return df
