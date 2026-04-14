"""
Input validation (range checks vs schema bounds).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ValidationResult:
    ok: bool
    warnings: list[str] = field(default_factory=list)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        self.ok = False


def validate_raw_row(
    row: dict[str, float] | pd.Series,
    bounds: dict[str, dict[str, float]],
) -> ValidationResult:
    """Validate one row of raw inputs (pre-engineering)."""
    r = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    res = ValidationResult(ok=True)

    for key, b in bounds.items():
        if key not in r:
            res.add_warning(f"Missing field: {key}")
            continue
        try:
            v = float(r[key])
        except (TypeError, ValueError):
            res.add_warning(f"{key} is not numeric")
            continue

        if key in ("chV", "disV") and v < 0:
            res.add_warning(f"{key} must be non-negative (got {v})")

        lo, hi = b["min"], b["max"]
        if v < lo or v > hi:
            res.add_warning(f"{key}={v} is outside typical range [{lo}, {hi}]")

    # Current sanity (magnitude)
    for k in ("chI", "disI"):
        if k in r and k in bounds and abs(float(r[k])) > 10:
            res.add_warning(f"{k} magnitude unusually large")

    return res


def validate_dataframe_raw(df: pd.DataFrame, bounds: dict[str, dict[str, float]]) -> list[ValidationResult]:
    """Per-row validation for batch CSV."""
    out: list[ValidationResult] = []
    for _, row in df.iterrows():
        out.append(validate_raw_row(row, bounds))
    return out
