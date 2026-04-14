# EV Battery Health Prediction (SOH)

## Description
Predict **battery SOH (State of Health, %)** from real EV cycling signals using an ML model, and **explain predictions with SHAP** (global + local).

## Cold Weather Range Degradation Modeler (NEW)
This repo also includes a **separate training pipeline** for your project topic:

- **Goal**: predict **range loss (%)** in cold weather by correlating **sub-zero ambient temperature** with **lithium-ion impedance** features.
- **Model**: XGBoost regressor
- **Leakage-safe split**: group split by `vehicle_id`/`battery_id` so the same vehicle/pack never appears in both train and test

### Dataset format (cold range)
Create `data/cold_weather_range_dataset.csv` with at least:

- **Group id**: `vehicle_id` (or `battery_id` / `pack_id`)
- **Inputs**:
  - `ambient_temp_c` (°C, include sub-zero rows)
  - `impedance_r0_mohm` (mΩ)
  - `impedance_rct_mohm` (mΩ)
- **Target (pick one option)**:
  - Option A: `range_loss_pct` (0–100)
  - Option B: `baseline_range_km` + `observed_range_km` (the script will compute `range_loss_pct`)

A small example template is included at `data/cold_weather_range_dataset_TEMPLATE.csv`.

### Train the cold range model

```bash
python train_cold_range.py
```

This creates:
- `model/xgb_cold_range_model.joblib`
- `model/cold_range_schema.json`

## Features
- **ML model**: XGBoost regressor for SOH prediction
- **Leakage-safe training**: group split by `battery_id` (no battery appears in both train and test)
- **SHAP explainability**: background sample + interactive Plotly visuals
- **Streamlit app**: single prediction UI + insights dashboard
- **Batch prediction**: upload CSV → download predictions

## Dataset note
This repo currently includes a **small dataset (3 batteries)**. Results can look optimistic because the test set may contain only 1 battery.

**WARNING: Small dataset — metrics may not generalize.**

## How to run
Install dependencies:

```bash
pip install -r requirements.txt
```

### Step 1 — Train

```bash
python train.py
```

This creates:
- `model/xgb_model.joblib`
- `model/schema.json` (feature order, bounds, and metrics used by the app)

### Step 2 — Run the app

```bash
streamlit run app.py
```

## Tech stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Streamlit, Joblib

