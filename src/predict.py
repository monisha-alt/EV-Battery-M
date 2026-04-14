"""
Safe model prediction with NaN/inf handling and SOH clamping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    soh: float
    abnormal: bool
    note: str


def safe_predict(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, list[PredictionResult]]:
    """
    Predict SOH (%) for each row; clamp to [0, 100], detect NaN/inf outputs.

    Returns (array of SOH values, per-row metadata).
    """
    if X.isna().any().any():
        raise ValueError("Input contains NaN — fix inputs before prediction.")

    raw = model.predict(X)
    arr = np.asarray(raw, dtype=float).reshape(-1)

    meta: list[PredictionResult] = []
    out = np.empty_like(arr)

    for i, v in enumerate(arr):
        abnormal = False
        notes: list[str] = []
        if not np.isfinite(v):
            abnormal = True
            notes.append("Model returned non-finite value; displayed as 0.")
            out[i] = 0.0
            meta.append(PredictionResult(soh=float(out[i]), abnormal=True, note=" ".join(notes)))
            continue

        v = float(v)
        if v < 0:
            abnormal = True
            notes.append("SOH below 0 clipped to 0.")
            v = 0.0

        if v > 100:
            abnormal = True
            notes.append("SOH above 100 clipped to 100.")
            v = 100.0

        rounded = float(np.round(v, 2))
        out[i] = rounded
        meta.append(PredictionResult(soh=rounded, abnormal=abnormal, note=" ".join(notes).strip()))

    return out, meta
