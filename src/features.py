"""
Feature engineering (TRAINING + INFERENCE must match).

Target: SOH (State of Health).
Raw columns are taken directly from the real dataset (no synthetic targets).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

# Strict training order (do not reorder)
FEATURE_ORDER: list[str] = [
    "cycle",
    "chI",
    "chV",
    "chT",
    "disI",
    "disV",
    "disT",
    # engineered from real columns
    "power_ch",  # chV * chI
    "power_dis",  # disV * disI
    "current_abs_ch",  # abs(chI)
    "current_abs_dis",  # abs(disI)
    "temp_diff",  # chT - disT
    "current_diff",  # chI - disI
    "voltage_diff",  # chV - disV
]

RAW_COLUMNS = ["cycle", "chI", "chV", "chT", "disI", "disV", "disT"]


def build_features(inputs: dict[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
    """
    Build engineered features from raw inputs.

    Accepts:
    - dict with keys: cycle, chI, chV, chT, disI, disV, disT
    - pandas Series (one row)
    - pandas DataFrame (multiple rows) with raw columns only

    Returns DataFrame with columns FEATURE_ORDER only.
    """
    if isinstance(inputs, pd.DataFrame):
        return _build_features_frame(inputs)

    if isinstance(inputs, pd.Series):
        row = inputs[RAW_COLUMNS].to_dict()
        return build_features(row)

    cycle = float(inputs["cycle"])
    chI = float(inputs["chI"])
    chV = float(inputs["chV"])
    chT = float(inputs["chT"])
    disI = float(inputs["disI"])
    disV = float(inputs["disV"])
    disT = float(inputs["disT"])

    power_ch = chV * chI
    power_dis = disV * disI
    current_abs_ch = abs(chI)
    current_abs_dis = abs(disI)
    temp_diff = chT - disT
    current_diff = chI - disI
    voltage_diff = chV - disV

    row = {
        "cycle": cycle,
        "chI": chI,
        "chV": chV,
        "chT": chT,
        "disI": disI,
        "disV": disV,
        "disT": disT,
        "power_ch": power_ch,
        "power_dis": power_dis,
        "current_abs_ch": current_abs_ch,
        "current_abs_dis": current_abs_dis,
        "temp_diff": temp_diff,
        "current_diff": current_diff,
        "voltage_diff": voltage_diff,
    }
    df = pd.DataFrame([row], columns=FEATURE_ORDER)
    return df


def _build_features_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized feature build for batch / CSV."""
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = pd.DataFrame(index=df.index)
    for c in RAW_COLUMNS:
        out[c] = pd.to_numeric(df[c], errors="coerce")

    out["power_ch"] = out["chV"] * out["chI"]
    out["power_dis"] = out["disV"] * out["disI"]
    out["current_abs_ch"] = out["chI"].abs()
    out["current_abs_dis"] = out["disI"].abs()
    out["temp_diff"] = out["chT"] - out["disT"]
    out["current_diff"] = out["chI"] - out["disI"]
    out["voltage_diff"] = out["chV"] - out["disV"]

    return out[FEATURE_ORDER].copy()
