from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

from src.cold_range_features import FEATURE_ORDER_COLD_RANGE, RAW_COLUMNS_COLD_RANGE, build_cold_range_features


@dataclass(frozen=True)
class TrainMetrics:
    r2: float
    mae: float
    rmse: float
    n_train: int
    n_test: int
    n_groups_train: int
    n_groups_test: int


def project_root() -> Path:
    return Path(__file__).resolve().parent


def _infer_group_col(df: pd.DataFrame) -> str:
    for c in ("vehicle_id", "battery_id", "pack_id"):
        if c in df.columns:
            return c
    raise ValueError('Missing group id column. Provide one of: "vehicle_id", "battery_id", "pack_id".')


def load_dataset(path: Path) -> pd.DataFrame:
    # Auto-detect delimiter (comma vs tab) for user-provided CSV/TSV exports.
    # Many Windows editors export as TSV even with a .csv extension.
    df = pd.read_csv(path, sep=None, engine="python")
    group_col = _infer_group_col(df)

    required_any_target = {"range_loss_pct"} | {"baseline_range_km", "observed_range_km"}
    required_inputs = {group_col, *RAW_COLUMNS_COLD_RANGE}

    missing_inputs = required_inputs - set(df.columns)
    if missing_inputs:
        raise ValueError(f"Dataset missing required columns: {sorted(missing_inputs)}")

    has_loss = "range_loss_pct" in df.columns
    has_ranges = "baseline_range_km" in df.columns and "observed_range_km" in df.columns
    if not (has_loss or has_ranges):
        raise ValueError(
            'Dataset must include either "range_loss_pct" '
            'or BOTH "baseline_range_km" and "observed_range_km".'
        )

    return df


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    group_col = _infer_group_col(out)

    # Inputs
    for c in RAW_COLUMNS_COLD_RANGE:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Target
    if "range_loss_pct" in out.columns:
        out["range_loss_pct"] = pd.to_numeric(out["range_loss_pct"], errors="coerce")
    else:
        out["baseline_range_km"] = pd.to_numeric(out["baseline_range_km"], errors="coerce")
        out["observed_range_km"] = pd.to_numeric(out["observed_range_km"], errors="coerce")
        base = out["baseline_range_km"]
        obs = out["observed_range_km"]
        out["range_loss_pct"] = (1.0 - (obs / base)).astype(float) * 100.0

    out[group_col] = out[group_col].astype(str)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[group_col, *RAW_COLUMNS_COLD_RANGE, "range_loss_pct"]).reset_index(drop=True)

    # Clamp to a sensible range for modeling; still keep raw labels in your source data.
    out["range_loss_pct"] = out["range_loss_pct"].clip(lower=0.0, upper=100.0)
    return out, group_col


def group_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    groups = df[group_col].astype(str).values
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return train_idx, test_idx


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, float, float, np.ndarray]:
    pred = model.predict(X_test).astype(float)
    r2 = float(r2_score(y_test, pred))
    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    return r2, mae, rmse, pred


def main() -> None:
    root = project_root()
    data_path = root / "data" / "cold_weather_range_dataset.csv"
    model_dir = root / "model"
    model_dir.mkdir(exist_ok=True)

    df = load_dataset(data_path)
    df, group_col = clean_dataframe(df)

    n_groups = int(df[group_col].nunique())
    print("\n=== Dataset validation (cold range) ===")
    print(f"Rows: {len(df)}")
    print(f"Unique {group_col}: {n_groups}")
    if n_groups < 10:
        print("WARNING: Small dataset — metrics may not generalize")

    train_idx, test_idx = group_split(df, group_col=group_col, test_size=0.2, random_state=42)
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    overlap = set(df_train[group_col].unique()) & set(df_test[group_col].unique())
    assert not overlap, f"Leakage detected: {group_col} overlap between train and test: {sorted(overlap)}"

    X_train = build_cold_range_features(df_train[RAW_COLUMNS_COLD_RANGE])
    y_train = df_train["range_loss_pct"].astype(float)
    X_test = build_cold_range_features(df_test[RAW_COLUMNS_COLD_RANGE])
    y_test = df_test["range_loss_pct"].astype(float)

    model = train_model(X_train, y_train)
    r2, mae, rmse, _pred = evaluate(model, X_test, y_test)

    metrics = TrainMetrics(
        r2=r2,
        mae=mae,
        rmse=rmse,
        n_train=int(len(df_train)),
        n_test=int(len(df_test)),
        n_groups_train=int(df_train[group_col].nunique()),
        n_groups_test=int(df_test[group_col].nunique()),
    )

    print("\n=== Cold Weather Range Degradation — XGBoost (group split) ===")
    print(f"Dataset: {data_path}")
    print(f"Train rows: {metrics.n_train:<5} | Test rows: {metrics.n_test:<5}")
    print(f"Train groups: {metrics.n_groups_train:<3} | Test groups: {metrics.n_groups_test:<3}")
    print("")
    print("=== Evaluation (test set) ===")
    print(f"R²   : {metrics.r2:8.4f}")
    print(f"MAE  : {metrics.mae:8.4f} % range loss")
    print(f"RMSE : {metrics.rmse:8.4f} % range loss")

    model_path = model_dir / "xgb_cold_range_model.joblib"
    joblib.dump(model, model_path)

    schema = {
        "model_version": "0.1.0",
        "model_type": "XGBoost",
        "prediction_target": "Range loss (%)",
        "group_column": group_col,
        "n_features": len(FEATURE_ORDER_COLD_RANGE),
        "feature_order": FEATURE_ORDER_COLD_RANGE,
        "raw_input_columns": RAW_COLUMNS_COLD_RANGE,
        "model_path": "model/xgb_cold_range_model.joblib",
        "dataset_path": "data/cold_weather_range_dataset.csv",
        "metrics": asdict(metrics),
        "notes": [
            'Provide impedance features in milli-ohms (mΩ) and ambient temperature in °C.',
            'Target is range loss (%) relative to baseline range at ~20°C.',
        ],
    }

    schema_path = model_dir / "cold_range_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print(f"\nSaved model: {model_path}")
    print(f"Saved schema: {schema_path}\n")


if __name__ == "__main__":
    main()

