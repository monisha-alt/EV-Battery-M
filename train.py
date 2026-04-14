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

from src.features import FEATURE_ORDER, RAW_COLUMNS, build_features


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


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"battery_id", *RAW_COLUMNS, "SOH"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in RAW_COLUMNS + ["SOH"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["battery_id", *RAW_COLUMNS, "SOH"]).reset_index(drop=True)
    return out


def group_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    groups = df["battery_id"].astype(str).values
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return train_idx, test_idx


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
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
    data_path = root / "data" / "Battery_dataset.csv"
    model_dir = root / "model"
    model_dir.mkdir(exist_ok=True)

    df = load_dataset(data_path)
    df = clean_dataframe(df)

    n_batteries = int(df["battery_id"].astype(str).nunique())
    print("\n=== Dataset validation ===")
    print(f"Rows: {len(df)}")
    print(f"Unique battery_id: {n_batteries}")
    if n_batteries < 10:
        print("WARNING: Small dataset — metrics may not generalize")

    train_idx, test_idx = group_split(df, test_size=0.2, random_state=42)
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    train_batteries = set(df_train["battery_id"].astype(str).unique())
    test_batteries = set(df_test["battery_id"].astype(str).unique())
    overlap = train_batteries & test_batteries
    assert not overlap, f"Leakage detected: battery_id overlap between train and test: {sorted(overlap)}"

    X_train = build_features(df_train[RAW_COLUMNS])
    y_train = df_train["SOH"].astype(float)
    X_test = build_features(df_test[RAW_COLUMNS])
    y_test = df_test["SOH"].astype(float)

    model = train_model(X_train, y_train)
    r2, mae, rmse, _pred = evaluate(model, X_test, y_test)

    metrics = TrainMetrics(
        r2=r2,
        mae=mae,
        rmse=rmse,
        n_train=int(len(df_train)),
        n_test=int(len(df_test)),
        n_groups_train=int(df_train["battery_id"].nunique()),
        n_groups_test=int(df_test["battery_id"].nunique()),
    )

    print("\n=== EV Battery Intelligence — SOH Model (Leakage-safe group split) ===")
    print(f"Dataset: {data_path}")
    print(f"Train rows: {metrics.n_train:<5} | Test rows: {metrics.n_test:<5}")
    print(f"Train batteries: {metrics.n_groups_train:<3} | Test batteries: {metrics.n_groups_test:<3}")
    print("")
    print("=== Evaluation (test set) ===")
    print(f"R²   : {metrics.r2:8.4f}")
    print(f"MAE  : {metrics.mae:8.4f} SOH%")
    print(f"RMSE : {metrics.rmse:8.4f} SOH%")

    model_path = model_dir / "xgb_model.joblib"
    joblib.dump(model, model_path)

    schema = {
        "model_version": "1.0.0",
        "model_type": "XGBoost",
        "prediction_target": "SOH (%)",
        "n_features": len(FEATURE_ORDER),
        "feature_order": FEATURE_ORDER,
        "raw_input_columns": RAW_COLUMNS,
        "model_search_paths": [
            "model/xgb_model.joblib",
            "xgb_model.joblib",
            "model/xgb_model.pkl",
            "xgb_model.pkl",
        ],
        "background_csv_paths": [
            "data/Battery_dataset.csv",
            "notebooks/Battery_dataset.csv",
        ],
        "background_sample_size": 300,
        "metrics": asdict(metrics),
        "bounds": {
            "cycle": {"min": int(df["cycle"].min()), "max": int(df["cycle"].max())},
            "chI": {"min": float(df["chI"].min()), "max": float(df["chI"].max())},
            "chV": {"min": float(df["chV"].min()), "max": float(df["chV"].max())},
            "chT": {"min": float(df["chT"].min()), "max": float(df["chT"].max())},
            "disI": {"min": float(df["disI"].min()), "max": float(df["disI"].max())},
            "disV": {"min": float(df["disV"].min()), "max": float(df["disV"].max())},
            "disT": {"min": float(df["disT"].min()), "max": float(df["disT"].max())},
        },
        "tooltips": {
            "cycle": "Cycle index since start of life.",
            "chI": "Charge current (A).",
            "chV": "Charge voltage (V).",
            "chT": "Temperature during charge (°C).",
            "disI": "Discharge current (A).",
            "disV": "Discharge voltage (V).",
            "disT": "Temperature during discharge (°C).",
        },
    }

    schema_path = model_dir / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print(f"\nSaved model: {model_path}")
    print(f"Saved schema: {schema_path}\n")


if __name__ == "__main__":
    main()

