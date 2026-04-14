"""
SHAP explainability: global (background) + local (waterfall via Plotly).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import plotly.express as px
except ImportError:  # pragma: no cover
    px = None

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

from src.features import FEATURE_ORDER, build_features


def load_background_frame(
    csv_paths: list[str],
    root: Path,
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load first available CSV, build features, sample rows for SHAP background."""
    path = None
    for rel in csv_paths:
        p = root / rel
        if p.is_file():
            path = p
            break
    if path is None:
        raise FileNotFoundError("No background CSV found for SHAP.")

    raw = pd.read_csv(path)
    need = ["cycle", "chI", "chV", "chT", "disI", "disV", "disT"]
    miss = set(need) - set(raw.columns)
    if miss:
        raise ValueError(f"Background CSV missing columns: {miss}")

    X = build_features(raw[need])
    n = min(sample_size, len(X))
    if n < 10:
        return X
    return X.sample(n=n, random_state=random_state).reset_index(drop=True)


def make_tree_explainer(model: Any, background: pd.DataFrame) -> Any:
    """TreeExplainer with background data for consistent SHAP estimates."""
    if shap is None:
        raise RuntimeError("shap is not installed")
    # Passing a background sample helps define an expected value baseline and
    # produces more stable attributions for tree models.
    return shap.TreeExplainer(model, data=background)


def explain_local(explainer: Any, X_row: pd.DataFrame) -> Any:
    """SHAP Explanation for a single row (same columns as training)."""
    return explainer(X_row)


def plotly_shap_waterfall(exp: Any, X_row: pd.DataFrame, row_index: int = 0) -> "go.Figure":
    """Interactive waterfall for one explanation (Plotly)."""
    if go is None:
        raise RuntimeError("plotly is not installed")

    vals = np.asarray(exp.values[row_index]).ravel()
    n = len(vals)
    names = list(X_row.columns)[:n]
    if len(names) != n:
        names = FEATURE_ORDER[:n]

    bv = exp.base_values
    if isinstance(bv, np.ndarray):
        base = float(bv[row_index] if bv.ndim > 0 else float(bv.flat[0]))
    else:
        base = float(bv)

    pred = base + float(vals.sum())

    # Order by |SHAP| so the most influential features appear first after the baseline
    order = np.argsort(np.abs(vals))[::-1]
    vals_o = vals[order]
    names_o = [names[i] for i in order]

    measure = ["absolute"] + ["relative"] * len(vals_o) + ["total"]
    x_labels = ["E[f(x)]"] + names_o + ["f(x)"]
    y_vals = [base] + list(vals_o) + [0.0]
    text_vals = [f"{base:.3g}"] + [f"{v:+.3g}" for v in vals_o] + [f"{pred:.3g}"]

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measure,
            x=x_labels,
            y=y_vals,
            text=text_vals,
            textposition="outside",
            connector={"line": {"color": "rgb(100,100,100)"}},
        )
    )
    fig.update_layout(
        title="Local SHAP (waterfall)",
        yaxis_title="Contribution toward SOH (%)",
        showlegend=False,
        height=520,
        margin=dict(t=50, b=100),
    )
    return fig


def plotly_global_mean_abs_shap(
    explainer: Any,
    X_sample: pd.DataFrame,
) -> "go.Figure":
    """Bar chart of mean |SHAP| over a background sample (global importance)."""
    if go is None:
        raise RuntimeError("plotly is not installed")

    exp = explainer(X_sample)
    vals = np.asarray(exp.values)
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    mean_abs = np.abs(vals).mean(axis=0)
    names = list(X_sample.columns)
    order = np.argsort(mean_abs)[::-1]

    fig = go.Figure(
        go.Bar(
            x=[names[i] for i in order],
            y=[mean_abs[i] for i in order],
            marker_color="steelblue",
        )
    )
    fig.update_layout(
        title="Global SHAP — mean |SHAP| (background sample)",
        xaxis_title="Feature",
        yaxis_title="mean |SHAP|",
        height=480,
        margin=dict(t=50, b=100),
    )
    return fig


def plotly_shap_summary_plot(
    explainer: Any,
    X_sample: pd.DataFrame,
    max_points_per_feature: int | None = 400,
) -> "go.Figure":
    """
    SHAP summary-style chart (Plotly): x = SHAP value, y = feature, color = feature value (per-feature min-max).
    Interactive alternative to matplotlib beeswarm summary.
    """
    if go is None or px is None:
        raise RuntimeError("plotly is not installed")

    exp = explainer(X_sample)
    sv = np.asarray(exp.values)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    n, m = sv.shape
    names = list(X_sample.columns)
    if getattr(exp, "data", None) is not None:
        feat = np.asarray(exp.data)
    else:
        feat = X_sample.values
    if feat.shape != sv.shape:
        feat = X_sample.values

    rows: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(m):
            rows.append(
                {
                    "shap": float(sv[i, j]),
                    "feature": names[j],
                    "fval": float(feat[i, j]),
                }
            )
    long_df = pd.DataFrame(rows)
    if max_points_per_feature and n * m > max_points_per_feature * m:
        long_df = long_df.sample(n=max_points_per_feature * m, random_state=42)

    long_df["fval_norm"] = long_df.groupby("feature", observed=True)["fval"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-12)
    )

    fig = px.scatter(
        long_df,
        x="shap",
        y="feature",
        color="fval_norm",
        color_continuous_scale="RdBu_r",
        range_color=(0, 1),
        height=max(420, 26 * m),
        labels={"shap": "SHAP value (impact on SOH)", "feature": "", "fval_norm": "Feature value (norm.)"},
    )
    fig.update_layout(
        title="SHAP summary plot (background sample)",
        yaxis={"categoryorder": "mean ascending"},
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    return fig


def top_features_from_shap(exp: Any, feature_names: list[str], row_index: int = 0, k: int = 3) -> str:
    """Dynamic one-line summary: top |SHAP| features for local instance."""
    vals = np.asarray(exp.values[row_index]).ravel()
    names = feature_names if len(feature_names) == len(vals) else FEATURE_ORDER[: len(vals)]
    order = np.argsort(np.abs(vals))[::-1][:k]
    parts = [f"{names[i]} ({vals[i]:+.3g})" for i in order]
    return "Largest local contributions: " + ", ".join(parts)
