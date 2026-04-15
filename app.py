"""
EV Battery Intelligence — Streamlit entrypoint.

Navigation: pending queue applied at startup so `nav_radio` is never written after the radio widget mounts.
"""
from __future__ import annotations

import html
import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import json

from model.loader import load_model, load_schema, project_root
from src.explain import (
    explain_local,
    load_background_frame,
    make_tree_explainer,
    plotly_global_mean_abs_shap,
    plotly_shap_summary_plot,
    plotly_shap_waterfall,
    top_features_from_shap,
)
from src.features import FEATURE_ORDER, RAW_COLUMNS, build_features
from src.cold_range_features import RAW_COLUMNS_COLD_RANGE, build_cold_range_features
from src.predict import safe_predict
from src.validation import validate_dataframe_raw, validate_raw_row

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

# -----------------------------------------------------------------------------
# Session-state keys (single source for sidebar radio)
# -----------------------------------------------------------------------------
NAV_RADIO_KEY = "nav_radio"
NAV_PENDING_KEY = "_nav_pending"  # consumed once at main() start — never bind to a widget

st.set_page_config(
    page_title="EV Battery Intelligence",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = project_root()

NAV_OPTIONS: list[tuple[str, str]] = [
    ("🏠 Home", "home"),
    ("⚡ Prediction", "prediction"),
    ("❄️ Cold Range", "cold_range"),
    ("📊 Insights", "insights"),
]
ID_BY_LABEL = {label: pid for label, pid in NAV_OPTIONS}
_LABELS = [x[0] for x in NAV_OPTIONS]


def _queue_nav(label: str) -> None:
    """Queue navigation target; applied on next run before any widget uses NAV_RADIO_KEY."""
    if label in ID_BY_LABEL:
        st.session_state[NAV_PENDING_KEY] = label


def _apply_pending_nav() -> None:
    """Move queued page into nav_radio before st.sidebar.radio is created."""
    if NAV_PENDING_KEY in st.session_state:
        pending = st.session_state.pop(NAV_PENDING_KEY)
        if pending in ID_BY_LABEL:
            st.session_state[NAV_RADIO_KEY] = pending
    if NAV_RADIO_KEY not in st.session_state:
        st.session_state[NAV_RADIO_KEY] = _LABELS[0]


@st.cache_resource
def get_cached_model():
    """One load per session — fast repeat predictions."""
    return load_model(schema=load_schema())


def load_cold_range_schema() -> dict:
    p = ROOT / "model" / "cold_range_schema.json"
    if not p.is_file():
        raise FileNotFoundError(f'Cold-range schema not found: "{p}". Run `python train_cold_range.py` first.')
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_resource
def get_cached_cold_range_model():
    schema = load_cold_range_schema()
    model_path = schema.get("model_path", "model/xgb_cold_range_model.joblib")
    return load_model(path=(ROOT / str(model_path)))


_DEFAULT_SLIDER_STATE = {
    "pred_cycle": 50,
    "pred_chI": 1.5,
    "pred_chV": 4.2,
    "pred_chT": 30,
    "pred_disI": 2.0,
    "pred_disV": 3.7,
    "pred_disT": 32,
}

_EXAMPLE_SLIDER_STATE = {
    "pred_cycle": 120,
    "pred_chI": 1.42,
    "pred_chV": 4.25,
    "pred_chT": 28,
    "pred_disI": 1.95,
    "pred_disV": 3.4,
    "pred_disT": 33,
}


def _reset_prediction_inputs() -> None:
    for k, v in _DEFAULT_SLIDER_STATE.items():
        st.session_state[k] = v


def _load_example_prediction_inputs() -> None:
    for k, v in _EXAMPLE_SLIDER_STATE.items():
        st.session_state[k] = v


@st.cache_data(show_spinner="Loading background data for SHAP…")
def load_bg_cached(paths_tuple: tuple[str, ...], sample_size: int) -> pd.DataFrame:
    return load_background_frame(list(paths_tuple), ROOT, sample_size)


def get_schema() -> dict:
    if "_schema" not in st.session_state:
        st.session_state["_schema"] = load_schema()
    return st.session_state["_schema"]


def inject_global_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

:root {
  /* Light theme */
  --accent-blue: #7c3aed;
  --accent-green: #14b8a6;

  --bg-main: radial-gradient(ellipse 110% 70% at 50% -20%, rgba(124, 58, 237, 0.10), transparent 60%),
    radial-gradient(ellipse 80% 60% at 85% 0%, rgba(20, 184, 166, 0.08), transparent 55%),
    linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
  --bg-sidebar: linear-gradient(180deg, rgba(248, 250, 252, 0.95) 0%, rgba(238, 242, 255, 0.95) 100%);

  --text-main: #0f172a;
  --text-soft: rgba(15, 23, 42, 0.86);
  --text-muted: rgba(15, 23, 42, 0.62);

  --card-bg: rgba(255, 255, 255, 0.82);
  --card-border: rgba(15, 23, 42, 0.12);
  --card-shadow: 0 12px 36px rgba(15, 23, 42, 0.12);
  --metric-bg: linear-gradient(145deg, rgba(255, 255, 255, 0.9), rgba(241, 245, 249, 0.92));

  --hero-bg: linear-gradient(135deg, rgba(124, 58, 237, 0.28) 0%, rgba(20, 184, 166, 0.18) 45%, rgba(244, 114, 182, 0.12) 100%),
    linear-gradient(135deg, #130f2f 0%, #0e1430 55%, #050814 100%);

  --toolbar-bg: rgba(124, 58, 237, 0.12);
  --toolbar-text: #ddd6fe;
  --toolbar-border: rgba(124, 58, 237, 0.35);

  --primary-bg: linear-gradient(135deg, #8b5cf6 0%, #14b8a6 100%);
  --primary-text: #f8fafc;

  --secondary-bg: rgba(231, 238, 252, 0.06);
  --secondary-text: rgba(231, 238, 252, 0.90);
  --secondary-border: rgba(255, 255, 255, 0.10);

  --result-bg: linear-gradient(180deg, rgba(255, 255, 255, 0.94) 0%, rgba(241, 245, 249, 0.96) 100%);
  --result-border: rgba(139, 92, 246, 0.32);
  --result-shadow: 0 16px 48px rgba(15, 23, 42, 0.16);
}

html, body, [class*="css"] {
  font-family: 'Inter', system-ui, sans-serif !important;
}

.main .block-container {
  padding-top: 1.2rem !important;
  padding-bottom: 1.6rem !important;
  padding-left: 1.25rem !important;
  padding-right: 1.25rem !important;
  max-width: 1120px !important;
}

section.main > div {
  gap: 0.7rem !important;
}

[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] .main {
  background: var(--bg-main) !important;
  color: var(--text-main) !important;
}

[data-testid="stSidebar"] {
  background: var(--bg-sidebar) !important;
  border-right: 1px solid var(--card-border);
  padding-top: 0.5rem;
}

[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
  color: var(--text-soft) !important;
}

.nav-section-title {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-muted) !important;
  margin-bottom: 0.25rem;
}

.sidebar-brand {
  border-radius: 16px;
  border: 1px solid rgba(124, 58, 237, 0.35);
  background: linear-gradient(145deg, rgba(124, 58, 237, 0.16), rgba(20, 184, 166, 0.10));
  padding: 12px 12px;
  margin-bottom: 0.65rem;
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.32);
}
.sidebar-brand h4 {
  margin: 0;
  color: #e9f4ff !important;
  font-size: 0.95rem;
  letter-spacing: -0.01em;
}
.sidebar-brand p {
  margin: 0.22rem 0 0 0;
  color: var(--text-muted) !important;
  font-size: 0.76rem;
}
.sidebar-chip-row {
  display: flex;
  gap: 0.35rem;
  flex-wrap: wrap;
  margin-top: 0.5rem;
}
.sidebar-chip {
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  padding: 0.24rem 0.55rem;
  background: rgba(15, 23, 42, 0.45);
  color: var(--text-soft) !important;
  font-size: 0.68rem;
  font-weight: 700;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label {
  border: 1px solid rgba(255, 255, 255, 0.09) !important;
  background: rgba(15, 23, 42, 0.45) !important;
  border-radius: 12px !important;
  padding: 0.5rem 0.62rem !important;
  margin-bottom: 0.4rem !important;
  transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
  transform: translateY(-1px);
  border-color: rgba(124, 58, 237, 0.55) !important;
  box-shadow: 0 8px 20px rgba(124, 58, 237, 0.16) !important;
}

.hero {
  background: var(--hero-bg);
  padding: 24px;
  border-radius: 20px;
  color: white;
  box-shadow: 0 22px 70px rgba(0, 0, 0, 0.45);
  margin-bottom: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  position: relative;
  overflow: hidden;
}
.hero:before {
  content: "";
  position: absolute;
  inset: -40%;
  background: radial-gradient(circle at 25% 25%, rgba(124, 58, 237, 0.30), transparent 55%),
    radial-gradient(circle at 75% 15%, rgba(20, 184, 166, 0.20), transparent 50%),
    radial-gradient(circle at 60% 85%, rgba(244, 114, 182, 0.14), transparent 55%);
  filter: blur(12px);
  opacity: 0.75;
  transform: rotate(8deg);
}
.hero > * { position: relative; z-index: 1; }
.hero h2 {
  margin: 0 0 0.35rem 0;
  font-size: 1.55rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #fff !important;
}
.hero p {
  margin: 0;
  font-size: 0.95rem;
  opacity: 0.95;
  line-height: 1.45;
}

/* TOOLBAR (under hero) */
.toolbar-row {
  display: flex;
  align-items: center;
  min-height: 2.5rem;
}
.toolbar-a {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  padding: 0.6rem 1rem;
  border-radius: 12px;
  font-weight: 600;
  font-size: 0.88rem;
  text-decoration: none !important;
  background: var(--toolbar-bg);
  color: var(--toolbar-text) !important;
  border: 1px solid var(--toolbar-border);
  transition: background 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.toolbar-a:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 26px rgba(124, 58, 237, 0.20);
  border-color: rgba(124, 58, 237, 0.50);
}

.card {
  background: var(--card-bg);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-radius: 18px;
  padding: 18px;
  margin-bottom: 14px;
  border: 1px solid var(--card-border);
  box-shadow: var(--card-shadow);
}
.card--tight {
  padding: 14px;
  margin-bottom: 10px;
}
.card h3, .card h4 {
  margin-top: 0;
  margin-bottom: 0.45rem;
}
.card p,
.card li,
.card label,
.card div {
  color: var(--text-soft);
}
.section-label {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin: 0 0 0.45rem 0;
}

.metric-card {
  background: var(--metric-bg);
  padding: 16px;
  border-radius: 16px;
  text-align: center;
  font-size: 0.8rem;
  color: var(--text-muted);
  line-height: 1.45;
  border: 1px solid var(--card-border);
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
}
.metric-card b {
  display: block;
  margin-top: 0.35rem;
  font-size: 1.02rem;
  font-weight: 800;
  color: var(--text-main);
}

.result-box {
  background: var(--result-bg);
  border-radius: 20px;
  padding: 24px 20px;
  text-align: left;
  box-shadow: var(--result-shadow);
  border: 1px solid var(--result-border);
}
.result-box h2, .result-box h3 {
  margin: 0 0 0.35rem 0;
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--text-soft) !important;
}
.result-box h1 {
  margin: 0.25rem 0 0.2rem 0;
  font-size: 3rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: var(--text-main) !important;
  line-height: 1.05;
}
.result-box .metric-unit {
  margin: 0.15rem 0 1rem 0;
  font-size: 0.88rem;
  color: var(--text-muted);
  opacity: 0.95;
}
.result-box p {
  margin: 0.15rem 0 0 0;
  font-size: 0.9rem;
  color: var(--text-soft);
}
.result-meta {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}
.result-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.55rem 0.8rem;
  border-radius: 999px;
  border: 1px solid var(--card-border);
  background: rgba(148, 163, 184, 0.08);
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-soft);
}
.status-dot {
  width: 0.62rem;
  height: 0.62rem;
  border-radius: 999px;
  display: inline-block;
}
.input-group-title {
  margin: 0 0 0.25rem 0;
  font-size: 1rem;
  font-weight: 700;
  color: var(--text-main);
}
.input-group-copy {
  margin: 0 0 0.85rem 0;
  font-size: 0.88rem;
  color: var(--text-muted);
}

.placeholder-result {
  text-align: left;
  padding: 1.75rem 0.75rem;
  color: var(--text-muted);
  font-size: 0.88rem;
}

.stCaption, [data-testid="stCaptionContainer"] {
  color: var(--text-muted) !important;
}

.stButton > button {
  border-radius: 12px !important;
  font-weight: 600 !important;
  padding: 0.62rem 1rem !important;
  min-height: 2.9rem !important;
  border: 1px solid transparent !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
}
.stButton > button[kind="primary"],
button[kind="primary"] {
  background: var(--primary-bg) !important;
  color: var(--primary-text) !important;
  border: none !important;
  box-shadow: 0 18px 44px rgba(124, 58, 237, 0.20) !important;
}
.stButton > button[kind="primary"]:hover,
button[kind="primary"]:hover {
  transform: translateY(-1px);
  box-shadow: 0 22px 56px rgba(20, 184, 166, 0.22) !important;
  filter: saturate(1.08);
}
.stButton > button[kind="secondary"],
button[kind="secondary"] {
  background: var(--secondary-bg) !important;
  color: var(--secondary-text) !important;
  border: 1px solid var(--secondary-border) !important;
}
.stButton > button[kind="secondary"]:hover,
button[kind="secondary"]:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 18px rgba(15, 23, 42, 0.08) !important;
}
[data-testid="stDownloadButton"] > button {
  background: var(--secondary-bg) !important;
  color: var(--secondary-text) !important;
  border: 1px solid var(--secondary-border) !important;
  border-radius: 12px !important;
}

hr {
  border: none;
  height: 1px;
  background: var(--card-border);
  margin: 0.5rem 0 0.65rem 0;
}

::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, var(--accent-blue), var(--accent-green));
  border-radius: 10px;
}

/* Sidebar slider thumb glow */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
  box-shadow: 0 0 0 6px rgba(0, 212, 255, 0.12) !important;
}

/* Subtle fade-in */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
.card, .result-box, .hero { animation: fadeInUp 240ms ease both; }

/* Futuristic cold-range dashboard */
.cold-title {
  margin: 0;
  font-size: 1.9rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: var(--text-main);
}
.cold-subtitle {
  margin: 0.2rem 0 0 0;
  color: var(--text-muted);
  font-size: 0.95rem;
}
.cold-divider {
  margin-top: 0.8rem;
  height: 2px;
  width: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(56, 189, 248, 0.85), rgba(34, 197, 94, 0.75), transparent);
}
.hero-prediction {
  border-radius: 22px;
  border: 1px solid rgba(56, 189, 248, 0.35);
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.86), rgba(2, 6, 23, 0.9));
  box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.08), 0 0 46px rgba(56, 189, 248, 0.12), 0 18px 50px rgba(0, 0, 0, 0.45);
  padding: 28px 24px;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.hero-prediction:after {
  content: "";
  position: absolute;
  inset: -50% auto auto -25%;
  width: 180%;
  height: 140%;
  background: radial-gradient(circle, rgba(56, 189, 248, 0.18), transparent 62%);
  pointer-events: none;
}
.hero-kicker {
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.75rem;
  color: var(--text-muted);
  font-weight: 700;
}
.hero-loss {
  margin: 0.5rem 0 0.1rem 0;
  font-size: 3.2rem;
  font-weight: 800;
  letter-spacing: -0.03em;
}
.hero-remaining {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--text-soft);
}
.cold-footnote {
  text-align: center;
  color: var(--text-muted);
  font-size: 0.85rem;
  margin-top: 0.6rem;
}
.card:hover {
  transform: translateY(-2px);
  transition: transform 0.2s ease;
}

[data-testid="stFileUploader"] {
  background: var(--card-bg);
  border: 1px dashed var(--secondary-border);
  border-radius: 16px;
  padding: 0.35rem;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _apply_plotly_dark(fig) -> None:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0f172a"),
    )


def _hero_block(schema: dict) -> None:
    """Hero banner — call only from Home."""
    ver = html.escape(str(schema.get("model_version", "1.0.0")))
    st.markdown(
        f"""
<div class="hero">
<h2>⚡ EV Battery Intelligence</h2>
<p>Predict battery life with ML + explainability</p>
<p style="margin:0.85rem 0 0;font-size:0.78rem;opacity:0.88;">Model v{ver}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_home_toolbar() -> None:
    """GitHub under hero (Home only)."""
    gh_url = "https://github.com"
    t1 = st.columns(1)[0]
    with t1:
        st.markdown(
            f'<div class="toolbar-row"><a class="toolbar-a" href="{gh_url}" target="_blank" rel="noopener">GitHub</a></div>',
            unsafe_allow_html=True,
        )


def render_compact_topbar(schema: dict) -> None:
    """Non-home pages: no hero — product line + actions."""
    ver = html.escape(str(schema.get("model_version", "1.0.0")))
    gh_url = "https://github.com"
    c1, c2 = st.columns([5, 1])
    with c1:
        st.caption(f"🔋 EV Battery Intelligence · v{ver}")
    with c2:
        st.markdown(
            f'<div class="toolbar-row" style="justify-content:flex-end;"><a class="toolbar-a" href="{gh_url}" target="_blank" rel="noopener">GitHub</a></div>',
            unsafe_allow_html=True,
        )


def sidebar_nav() -> str:
    st.sidebar.markdown(
        """
<div class="sidebar-brand">
  <h4>⚡ EV Intelligence Console</h4>
  <p>Realtime battery analytics dashboard</p>
  <div class="sidebar-chip-row">
    <span class="sidebar-chip">AI</span>
    <span class="sidebar-chip">XGBoost</span>
    <span class="sidebar-chip">SHAP</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<p class="nav-section-title">Navigation</p>', unsafe_allow_html=True)
    choice = st.sidebar.radio(
        "nav_inner",
        _LABELS,
        key=NAV_RADIO_KEY,
        label_visibility="collapsed",
    )
    st.sidebar.markdown('<p class="nav-section-title" style="margin-top:0.6rem;">Tips</p>', unsafe_allow_html=True)
    st.sidebar.caption("Use `❄️ Cold Range` for the futuristic predictor view.")
    return ID_BY_LABEL[choice]


def init_input_state() -> None:
    for k, v in _DEFAULT_SLIDER_STATE.items():
        if k not in st.session_state:
            st.session_state[k] = v


def feature_importance_df(model) -> pd.DataFrame | None:
    if not hasattr(model, "feature_importances_"):
        return None
    imp = np.asarray(model.feature_importances_).ravel()
    if len(imp) != len(FEATURE_ORDER):
        return None
    df = pd.DataFrame({"Feature": FEATURE_ORDER, "Importance": imp})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


def ensure_explainer(model, schema: dict):
    key = "_tree_explainer"
    bg_key = "_shap_background"
    paths = tuple(schema["background_csv_paths"])
    n = int(schema.get("background_sample_size", 300))
    try:
        bg = align_feature_frame(model, load_bg_cached(paths, n))
    except Exception:
        return None, None
    if key not in st.session_state or st.session_state.get("_explainer_model_id") != id(model):
        if shap is None:
            return None, bg
        st.session_state[key] = make_tree_explainer(model, bg)
        st.session_state[bg_key] = bg
        st.session_state["_explainer_model_id"] = id(model)
    return st.session_state[key], st.session_state.get(bg_key)


def get_model_feature_names(model) -> list[str] | None:
    if not hasattr(model, "get_booster"):
        return None
    try:
        booster = model.get_booster()
    except Exception:
        return None
    feature_names = getattr(booster, "feature_names", None)
    return list(feature_names) if feature_names else None


def align_feature_frame(model, X: pd.DataFrame) -> pd.DataFrame:
    expected_names = get_model_feature_names(model)
    if not expected_names:
        return X
    missing_feats = [name for name in expected_names if name not in X.columns]
    extra_feats = [name for name in X.columns if name not in expected_names]
    if missing_feats or extra_feats:
        raise ValueError(
            "Feature mismatch with trained model. "
            f"Missing: {missing_feats or 'None'} | Extra: {extra_feats or 'None'}"
        )
    return X.reindex(columns=expected_names)


def soh_status(soh: float) -> tuple[str, str]:
    if soh > 90:
        return "Healthy", "#22c55e"
    if soh >= 70:
        return "Moderate", "#f59e0b"
    return "Degraded", "#ef4444"


def page_home() -> None:
    schema = get_schema()
    _hero_block(schema)
    render_home_toolbar()

    st.markdown("### 🔹 Dashboard")
    st.caption("Snapshot of the deployed model, inputs, and quick entry points.")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    mt = html.escape(str(schema.get("model_type", "XGBoost")))
    nf = html.escape(str(schema.get("n_features", len(FEATURE_ORDER))))
    tg = html.escape(str(schema.get("prediction_target", "SOH (%)")))
    vr = html.escape(str(schema.get("model_version", "—")))
    metrics = schema.get("metrics", {}) if isinstance(schema, dict) else {}
    r2_val = metrics.get("r2", None)
    mae_val = metrics.get("mae", None)
    rmse_val = metrics.get("rmse", None)
    r2 = html.escape(f"{float(r2_val):.3f}" if r2_val is not None else "—")
    mae = html.escape(f"{float(mae_val):.3f}" if mae_val is not None else "—")
    rmse = html.escape(f"{float(rmse_val):.3f}" if rmse_val is not None else "—")
    with m1:
        st.markdown(
            f'<div class="metric-card">🧠 Model<br><b>{mt}</b></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card">📊 Features<br><b>{nf}</b></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card">🎯 Target<br><b>{tg}</b></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-card">📈 R² Score<br><b>{r2}</b></div>',
            unsafe_allow_html=True,
        )
    with m5:
        st.markdown(
            f'<div class="metric-card">📉 MAE<br><b>{mae}</b></div>',
            unsafe_allow_html=True,
        )
    with m6:
        st.markdown(
            f'<div class="metric-card">📐 RMSE<br><b>{rmse}</b></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 🔹 Quick actions")
    q1, q2 = st.columns(2)
    with q1:
        st.button(
            "⚡ Go to Prediction",
            type="primary",
            use_container_width=True,
            key="qa_pred",
            on_click=_queue_nav,
            args=("⚡ Prediction",),
        )
    with q2:
        st.button(
            "📁 Batch (CSV)",
            type="secondary",
            use_container_width=True,
            key="qa_csv",
            on_click=_queue_nav,
            args=("⚡ Prediction",),
        )

    st.markdown("### 🔹 At a glance")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
<div class="card card--tight">
<h3 style="margin-top:0;">Capabilities</h3>
<ul style="margin:0;padding-left:1.1rem;line-height:1.5;font-size:0.92rem;">
<li><b>⚡ Prediction</b> — sliders + input validation</li>
<li><b>📁 Batch</b> — CSV upload &amp; download</li>
<li><b>📊 Insights</b> — importance + SHAP</li>
<li><b>ℹ️ About</b> — artifacts &amp; scope</li>
</ul>
</div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div class="card card--tight">
<h3 style="margin-top:0;">Workflow</h3>
<ol style="margin:0;padding-left:1.1rem;line-height:1.5;font-size:0.92rem;">
<li><b>⚡ Prediction</b> → tune → <b>Predict SOH</b></li>
<li><b>📊 Insights</b> → global / local SHAP</li>
<li><b>📁 Batch</b> for many rows</li>
</ol>
</div>
""",
            unsafe_allow_html=True,
        )


def input_sliders(schema: dict) -> tuple:
    tips = schema.get("tooltips", {})
    with st.container(border=True):
        st.markdown('<p class="section-label">⚡ Charge signals</p>', unsafe_allow_html=True)
        st.markdown('<p class="input-group-title">Charging profile</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="input-group-copy">Set cycle index and charge-side operating conditions.</p>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2, gap="large")
        with c1:
            cycle = st.slider("Cycle count", 0, 300, key="pred_cycle", help=tips.get("cycle", "Current battery cycle number used for feature engineering."))
            chI = st.slider("Charge current (A)", 1.0, 2.0, 0.01, key="pred_chI", help=tips.get("chI", "Average charging current in amperes."))
        with c2:
            chV = st.slider("Charge voltage (V)", 4.0, 4.5, 0.01, key="pred_chV", help=tips.get("chV", "Observed charging voltage in volts."))
            chT = st.slider("Charge temperature (°C)", 20, 40, 1, key="pred_chT", help=tips.get("chT", "Battery temperature during charging."))

    st.write("")

    with st.container(border=True):
        st.markdown('<p class="section-label">🔋 Discharge signals</p>', unsafe_allow_html=True)
        st.markdown('<p class="input-group-title">Discharging profile</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="input-group-copy">Capture load-side current, voltage, and thermal behavior.</p>',
            unsafe_allow_html=True,
        )
        c3, c4 = st.columns(2, gap="large")
        with c3:
            disI = st.slider("Discharge current (A)", 1.0, 3.0, 0.01, key="pred_disI", help=tips.get("disI", "Average discharge current in amperes."))
            disV = st.slider("Discharge voltage (V)", 2.5, 4.5, 0.01, key="pred_disV", help=tips.get("disV", "Observed discharge voltage in volts."))
        with c4:
            disT = st.slider("Discharge temperature (°C)", 25, 40, 1, key="pred_disT", help=tips.get("disT", "Battery temperature during discharge."))
    return cycle, chI, chV, chT, disI, disV, disT


def page_prediction() -> None:
    schema = get_schema()
    bounds = schema["bounds"]
    init_input_state()

    st.markdown("## ⚡ Prediction")
    st.caption("Tune charge/discharge inputs, run a single prediction, or batch a CSV. Last single run feeds **📊 Insights → Local SHAP**.")

    st.markdown("### 🔹 Single run")
    b1, b2 = st.columns(2)
    with b1:
        st.button(
            "↺ Reset inputs",
            type="secondary",
            use_container_width=True,
            key="btn_reset",
            on_click=_reset_prediction_inputs,
        )
    with b2:
        st.button(
            "✨ Load example",
            type="secondary",
            use_container_width=True,
            key="btn_example",
            on_click=_load_example_prediction_inputs,
        )

    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ⚡ Battery inputs")
        cycle, chI, chV, chT, disI, disV, disT = input_sliders(schema)
        raw = {
            "cycle": float(cycle),
            "chI": float(chI),
            "chV": float(chV),
            "chT": float(chT),
            "disI": float(disI),
            "disV": float(disV),
            "disT": float(disT),
        }
        vr = validate_raw_row(raw, bounds)
        if vr.warnings:
            for w in vr.warnings:
                st.warning(w)

        run = st.button("⚡ Predict SOH", type="primary", use_container_width=True, key="predict_main")
        if run:
            try:
                with st.spinner("🔍 Analyzing battery signals..."):
                    model = get_cached_model()
                    X = align_feature_frame(model, build_features(raw))
                    preds, metas = safe_predict(model, X)
                    soh = float(preds[0])
                    pm = metas[0]

                    st.session_state["last_input_df"] = X
                    st.session_state["last_prediction"] = soh
                    st.session_state["last_pred_note"] = pm.note
            except FileNotFoundError as e:
                st.error(str(e))
                st.info("Run training first to create `model/xgb_model.joblib` (see README / About).")
            except Exception as e:  # pragma: no cover
                st.error("Prediction failed.")
                st.exception(e)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ⚡ Result")
        if "last_prediction" in st.session_state:
            soh = float(st.session_state["last_prediction"])
            note_pm = st.session_state.get("last_pred_note", "")
            status_label, status_color = soh_status(soh)

            st.markdown(
                f"""
<div class="result-box">
  <h3>🔋 Predicted SOH</h3>
  <h1 style="font-size:48px;line-height:1.1;">{soh:.2f}%</h1>
  <p class="metric-unit">state of health</p>
  <div class="result-meta">
    <span class="result-pill"><span class="status-dot" style="background:{status_color};"></span>Status: {status_label}</span>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            if note_pm:
                st.caption(f"⚠ {note_pm}")
        else:
            st.info("Run **Predict SOH** on the left to see your estimate here.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🔹 Batch")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📁 Batch Prediction")
    st.caption(f"Required columns: {', '.join(RAW_COLUMNS)}")
    up = st.file_uploader("Upload CSV with raw signals", type=["csv"], key="batch_csv")
    if up is not None:
        try:
            df_raw = pd.read_csv(up)

            st.markdown("##### Preview")
            st.dataframe(df_raw.head(10), use_container_width=True)

            miss = set(RAW_COLUMNS) - set(df_raw.columns)
            if miss:
                st.error(f"Missing required columns: {sorted(miss)}")
            else:
                # Coerce to numeric and handle non-numeric / missing values.
                df_proc = df_raw.copy()
                for col in RAW_COLUMNS:
                    df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce")

                before_rows = len(df_proc)
                df_proc = df_proc.dropna(subset=RAW_COLUMNS)
                dropped = before_rows - len(df_proc)
                if dropped > 0:
                    st.warning(
                        f"Dropped {dropped} rows due to missing or non-numeric values in required columns."
                    )

                if df_proc.empty:
                    st.error("No valid rows left after cleaning; please check your CSV.")
                else:
                    model = get_cached_model()
                    with st.spinner("Running batch prediction..."):
                        Xb = align_feature_frame(model, build_features(df_proc[RAW_COLUMNS]))
                        vrs = validate_dataframe_raw(df_proc[RAW_COLUMNS], bounds)
                        preds, metas = safe_predict(model, Xb)

                        out = df_proc.copy()
                        out["Predicted_SOH"] = preds
                        out["Warnings"] = ["; ".join(vrs[i].warnings) if vrs[i].warnings else "" for i in range(len(vrs))]
                        out["Notes"] = [m.note for m in metas]

                    st.markdown("##### Results (first 50 rows)")
                    st.dataframe(out.head(50), use_container_width=True)

                    buf = io.StringIO()
                    out.to_csv(buf, index=False)
                    st.download_button(
                        "⬇️ Download results CSV",
                        buf.getvalue(),
                        file_name="soh_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error("Batch prediction failed.")
            st.exception(e)
    else:
        st.info("Upload a CSV to run batch predictions.")
    st.markdown("</div>", unsafe_allow_html=True)


def page_cold_range() -> None:
    st.markdown('<h2 class="cold-title">⚡ EV Cold Weather Range Intelligence</h2>', unsafe_allow_html=True)
    st.markdown('<p class="cold-subtitle">AI-powered battery performance insights</p>', unsafe_allow_html=True)
    st.markdown('<div class="cold-divider"></div>', unsafe_allow_html=True)

    try:
        cold_model = get_cached_cold_range_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Train the model first with `python train_cold_range.py`.")
        return

    if "cold_soc_pct" not in st.session_state:
        st.session_state["cold_soc_pct"] = 80
    if "cold_speed_kph" not in st.session_state:
        st.session_state["cold_speed_kph"] = 65
    if "cold_ambient_temp_c" not in st.session_state:
        st.session_state["cold_ambient_temp_c"] = -10
    if "cold_resistance_mohm" not in st.session_state:
        st.session_state["cold_resistance_mohm"] = 72

    left_col, right_col = st.columns([1, 1.7], gap="large")
    with left_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Input Controls")
        temp_c = st.slider("🌡️ Temperature (°C)", -40, 20, key="cold_ambient_temp_c")
        soc_pct = st.slider("🔋 SOC (%)", 10, 100, key="cold_soc_pct")
        r0 = st.slider("⚙️ Resistance (mΩ)", 30, 180, key="cold_resistance_mohm")
        speed_kph = st.slider("🚗 Speed (km/h)", 20, 130, key="cold_speed_kph")
        st.caption("Rct is estimated automatically from temperature + resistance trend.")
        if st.button("⚡ Run AI Prediction", type="primary", use_container_width=True, key="cold_predict_main"):
            st.session_state["cold_do_predict"] = True
        st.markdown("</div>", unsafe_allow_html=True)

    rct = float(r0) * (2.0 + max(0.0, -float(temp_c)) / 42.0)
    raw = {
        "ambient_temp_c": float(temp_c),
        "impedance_r0_mohm": float(r0),
        "impedance_rct_mohm": float(rct),
    }
    if st.session_state.get("cold_do_predict", True):
        with st.spinner("Running AI inference..."):
            X = align_feature_frame(cold_model, build_cold_range_features(raw))
            preds, metas = safe_predict(cold_model, X)
            st.session_state["cold_last_prediction"] = float(preds[0])
            st.session_state["cold_last_pred_note"] = metas[0].note
            st.session_state["cold_last_input_df"] = X
            st.session_state["cold_do_predict"] = False

    loss = float(st.session_state.get("cold_last_prediction", 0.0))
    note_pm = st.session_state.get("cold_last_pred_note", "")
    nominal_range_km = 430.0
    speed_factor = float(np.clip(1.0 - max(0.0, speed_kph - 70) * 0.0035, 0.72, 1.03))
    remaining_range_km = nominal_range_km * (soc_pct / 100.0) * (1.0 - loss / 100.0) * speed_factor

    if loss <= 15:
        severity_label, severity_color = "LOW LOSS", "#22c55e"
    elif loss <= 35:
        severity_label, severity_color = "MODERATE LOSS", "#f59e0b"
    else:
        severity_label, severity_color = "HIGH LOSS", "#ef4444"

    st.markdown(
        f"""
<div class="hero-prediction">
  <div class="hero-kicker">{severity_label}</div>
  <div class="hero-loss" style="color:{severity_color};">Predicted Range Loss: {loss:.1f}%</div>
  <p class="hero-remaining">Remaining Range: {remaining_range_km:.0f} km</p>
</div>
""",
        unsafe_allow_html=True,
    )
    if note_pm:
        st.caption(f"⚠ {note_pm}")

    with right_col:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 📉 Temperature vs Range Loss")
            temp_sweep = np.arange(-40, 21, 5, dtype=float)
            rct_sweep = float(r0) * (2.0 + np.maximum(0.0, -temp_sweep) / 42.0)
            sweep_df = pd.DataFrame(
                {
                    "ambient_temp_c": temp_sweep,
                    "impedance_r0_mohm": float(r0),
                    "impedance_rct_mohm": rct_sweep,
                }
            )
            X_sweep = align_feature_frame(cold_model, build_cold_range_features(sweep_df))
            y_sweep, _ = safe_predict(cold_model, X_sweep)
            fig_temp = px.line(
                x=temp_sweep,
                y=y_sweep,
                labels={"x": "Temperature (°C)", "y": "Range loss (%)"},
                markers=True,
                color_discrete_sequence=["#38bdf8"],
            )
            fig_temp.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
            _apply_plotly_dark(fig_temp)
            st.plotly_chart(fig_temp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🧠 Feature Importance")
            if hasattr(cold_model, "feature_importances_"):
                names = list(build_cold_range_features(raw).columns)
                imp = np.asarray(cold_model.feature_importances_, dtype=float)
                imp_df = pd.DataFrame({"Feature": names[: len(imp)], "Importance": imp}).sort_values("Importance", ascending=False)
                fig_imp = px.bar(
                    imp_df.head(10),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color_discrete_sequence=["#22c55e"],
                )
                fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, height=320, margin=dict(l=10, r=10, t=30, b=10))
                _apply_plotly_dark(fig_imp)
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Model-native feature importance is unavailable.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🔬 Advanced Insights")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Top influencing factors")
    x_local = st.session_state.get("cold_last_input_df")
    if x_local is None:
        x_local = align_feature_frame(cold_model, build_cold_range_features(raw))

    if shap is not None:
        try:
            explainer = make_tree_explainer(cold_model, x_local)
            exp = explain_local(explainer, x_local)
            summary = top_features_from_shap(exp, list(x_local.columns), row_index=0, k=5)
            st.success(summary)
            shap_vals = np.abs(exp.values[0]).astype(float)
            top_idx = np.argsort(shap_vals)[-6:][::-1]
            top_df = pd.DataFrame(
                {"Feature": [x_local.columns[i] for i in top_idx], "Influence": [shap_vals[i] for i in top_idx]}
            )
            fig_top = px.bar(top_df, x="Influence", y="Feature", orientation="h", color_discrete_sequence=["#38bdf8"])
            fig_top.update_layout(yaxis={"categoryorder": "total ascending"}, height=320, margin=dict(l=10, r=10, t=20, b=10))
            _apply_plotly_dark(fig_top)
            st.plotly_chart(fig_top, use_container_width=True)
        except Exception:
            st.info("SHAP is available but local factor extraction failed for this run.")
    else:
        st.info("Install `shap` to enable local explainability.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<p class="cold-footnote">Built with Machine Learning & XGBoost</p>', unsafe_allow_html=True)


def page_insights() -> None:
    schema = get_schema()
    st.markdown("## 📊 Insights")
    st.caption("Model importance, SHAP summary, global |SHAP|, and local waterfall — tied to your last single prediction when available.")

    try:
        model = get_cached_model()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    st.markdown("### 🔹 Model snapshot")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Type", schema.get("model_type", "—"))
    with m2:
        st.metric("Features", len(FEATURE_ORDER))
    with m3:
        st.metric("Target", schema.get("prediction_target", "SOH (%)"))
    with m4:
        st.metric("Version", schema.get("model_version", "—"))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🔹 Global feature importance (model)")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    imp_df = feature_importance_df(model)
    if imp_df is None:
        st.info("Feature importance not available for this model (missing or mismatched `feature_importances_`).")
    else:
        fig = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="XGBoost feature importances",
            color_discrete_sequence=["#818cf8"],
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
        _apply_plotly_dark(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("These features contribute most to tree splits (model-native importance).")
    st.markdown("</div>", unsafe_allow_html=True)

    if shap is None:
        st.warning("Install `shap` for SHAP charts (`pip install shap`).")
        return

    st.markdown("### 🔹 SHAP analysis")
    explainer, bg = ensure_explainer(model, schema)
    if explainer is None:
        st.warning("Could not build SHAP explainer (missing background CSV).")
        return

    X_loc = st.session_state.get("last_input_df")
    if X_loc is None:
        X_loc = align_feature_frame(
            model,
            build_features(
                {
                    "cycle": 50.0,
                    "chI": 1.5,
                    "chV": 4.2,
                    "chT": 30.0,
                    "disI": 2.0,
                    "disV": 3.7,
                    "disT": 32.0,
                }
            ),
        )

    tab_global, tab_local = st.tabs(["🌍 Global", "📍 Local"])

    with tab_global:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Summary plot")
        try:
            fig_s = plotly_shap_summary_plot(explainer, bg)
            _apply_plotly_dark(fig_s)
            st.plotly_chart(fig_s, use_container_width=True)
            st.caption("SHAP value vs feature; color = normalized feature value (background sample).")
        except Exception as e:  # pragma: no cover
            st.warning(f"SHAP summary plot failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Mean |SHAP| (global)")
        try:
            fig_g = plotly_global_mean_abs_shap(explainer, bg)
            _apply_plotly_dark(fig_g)
            fig_g.update_traces(marker_color="#818cf8")
            st.plotly_chart(fig_g, use_container_width=True)
            st.caption("Average |SHAP| over the background sample.")
        except Exception as e:  # pragma: no cover
            st.warning(f"Global SHAP bar failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_local:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Waterfall (local)")
        if st.session_state.get("last_input_df") is None:
            st.info("No prediction in this session — showing default input. Run **Prediction** first for your row.")
        try:
            exp = explain_local(explainer, X_loc)
            fig_w = plotly_shap_waterfall(exp, X_loc, row_index=0)
            _apply_plotly_dark(fig_w)
            st.plotly_chart(fig_w, use_container_width=True)
            summary = top_features_from_shap(exp, list(X_loc.columns), row_index=0, k=3)
            st.success(summary)
            st.caption("How each feature pushes the prediction up or down for this input.")
        except Exception as e:  # pragma: no cover
            st.warning(f"Local SHAP failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    # 1) Navigation: apply queued page BEFORE any widget reads NAV_RADIO_KEY
    _apply_pending_nav()

    inject_global_css()
    schema = get_schema()
    page = sidebar_nav()
    if page != "home":
        render_compact_topbar(schema)

    if page == "home":
        page_home()
    elif page == "prediction":
        page_prediction()
    elif page == "cold_range":
        page_cold_range()
    elif page == "insights":
        page_insights()


if __name__ == "__main__":
    main()
