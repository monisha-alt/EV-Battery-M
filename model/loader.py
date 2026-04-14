"""
Load trained estimators with joblib (preferred) and legacy pickle fallback.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import joblib


def project_root() -> Path:
    """Directory containing app.py (parent of model/)."""
    return Path(__file__).resolve().parent.parent


def load_schema(path: str | Path | None = None) -> dict[str, Any]:
    """Load model/schema.json (feature order, paths, bounds)."""
    if path is None:
        path = project_root() / "model" / "schema.json"
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Schema not found: "{p}"')
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_model(path: str | Path | None = None, schema: dict[str, Any] | None = None) -> Any:
    """
    Load XGBoost/sklearn model with joblib first, then pickle for legacy .pkl dumps.

    If path is None, uses schema model_search_paths.
    """
    root = project_root()
    if path is not None:
        candidates = [Path(path)]
    else:
        s = schema or load_schema()
        candidates = [root / p for p in s["model_search_paths"]]

    last_error: Exception | None = None
    for fp in candidates:
        if not fp.is_file():
            continue
        try:
            # Primary: joblib (recommended for sklearn/xgboost)
            return joblib.load(fp)
        except Exception as e1:
            last_error = e1
            try:
                with open(fp, "rb") as f:
                    return pickle.load(f)
            except Exception as e2:
                last_error = e2
                continue

    msg = "No loadable model file found."
    if last_error:
        msg += f" Last error: {last_error}"
    raise FileNotFoundError(msg)
