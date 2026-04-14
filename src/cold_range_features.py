"""
Cold Weather Range Degradation — feature engineering.

This module is intentionally separate from the SOH pipeline so the repo can
support both:
- SOH (%) prediction from cycling signals (existing)
- Range loss (%) prediction from cold ambient temperature + impedance (new)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Raw columns expected for training/inference (minimum viable schema).
RAW_COLUMNS_COLD_RANGE: list[str] = [
    "ambient_temp_c",
    "impedance_r0_mohm",
    "impedance_rct_mohm",
]

# Model input order (do not reorder once a model is trained).
FEATURE_ORDER_COLD_RANGE: list[str] = [
    *RAW_COLUMNS_COLD_RANGE,
    # engineered
    "temp_is_subzero",
    "temp_below_0_c",
    "temp_below_10_c",
    "temp_sq",
    "ln_r0",
    "ln_rct",
    "rct_over_r0",
    "temp_x_r0",
    "temp_x_rct",
]


def build_cold_range_features(inputs: dict[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
    """
    Build features for cold-weather range loss model.

    Accepts:
    - dict with keys in RAW_COLUMNS_COLD_RANGE
    - pandas Series (one row)
    - pandas DataFrame (multiple rows) with RAW_COLUMNS_COLD_RANGE

    Returns DataFrame with columns FEATURE_ORDER_COLD_RANGE only.
    """
    if isinstance(inputs, pd.DataFrame):
        return _build_frame(inputs)
    if isinstance(inputs, pd.Series):
        row = inputs[RAW_COLUMNS_COLD_RANGE].to_dict()
        return build_cold_range_features(row)

    t = float(inputs["ambient_temp_c"])
    r0 = float(inputs["impedance_r0_mohm"])
    rct = float(inputs["impedance_rct_mohm"])

    row = _engineer_row(t=t, r0=r0, rct=rct)
    return pd.DataFrame([row], columns=FEATURE_ORDER_COLD_RANGE)


def _engineer_row(*, t: float, r0: float, rct: float) -> dict[str, float]:
    eps = 1e-9
    temp_is_subzero = 1.0 if t < 0.0 else 0.0
    temp_below_0_c = float(max(0.0, -t))  # severity below freezing
    temp_below_10_c = float(max(0.0, -10.0 - t))  # severity below -10C
    temp_sq = float(t * t)

    ln_r0 = float(np.log(max(r0, eps)))
    ln_rct = float(np.log(max(rct, eps)))
    rct_over_r0 = float(rct / max(r0, eps))

    temp_x_r0 = float(t * r0)
    temp_x_rct = float(t * rct)

    return {
        "ambient_temp_c": float(t),
        "impedance_r0_mohm": float(r0),
        "impedance_rct_mohm": float(rct),
        "temp_is_subzero": temp_is_subzero,
        "temp_below_0_c": temp_below_0_c,
        "temp_below_10_c": temp_below_10_c,
        "temp_sq": temp_sq,
        "ln_r0": ln_r0,
        "ln_rct": ln_rct,
        "rct_over_r0": rct_over_r0,
        "temp_x_r0": temp_x_r0,
        "temp_x_rct": temp_x_rct,
    }


def _build_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(RAW_COLUMNS_COLD_RANGE) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    x = df.copy()
    for c in RAW_COLUMNS_COLD_RANGE:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    out = pd.DataFrame(index=x.index)
    out["ambient_temp_c"] = x["ambient_temp_c"].astype(float)
    out["impedance_r0_mohm"] = x["impedance_r0_mohm"].astype(float)
    out["impedance_rct_mohm"] = x["impedance_rct_mohm"].astype(float)

    t = out["ambient_temp_c"]
    r0 = out["impedance_r0_mohm"]
    rct = out["impedance_rct_mohm"]
    eps = 1e-9

    out["temp_is_subzero"] = (t < 0.0).astype(float)
    out["temp_below_0_c"] = (-t).clip(lower=0.0)
    out["temp_below_10_c"] = (-10.0 - t).clip(lower=0.0)
    out["temp_sq"] = t * t

    out["ln_r0"] = np.log(r0.clip(lower=eps))
    out["ln_rct"] = np.log(rct.clip(lower=eps))
    out["rct_over_r0"] = rct / r0.clip(lower=eps)

    out["temp_x_r0"] = t * r0
    out["temp_x_rct"] = t * rct

    return out[FEATURE_ORDER_COLD_RANGE].copy()

