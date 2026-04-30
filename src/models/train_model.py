"""
train_model.py
--------------
Trains a GradientBoostingRegressor on the unified features.csv.
Uses a TIME-BASED train/test split (no leakage).
Saves both model and scaler to artifacts/.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import json
from datetime import datetime


FEATURE_COLS = [
    "driver_form_last5",
    "constructor_form_last5",
    "grid_form_last5",
    "pit_form_last5",
    "lap_consistency_last5",
    "circuit_avg_points",
    "cum_season_points",
    "grid",
]

TARGET_COL = "points"


def time_based_split(df, test_seasons=1):
    """
    Split by season to avoid leakage.
    Train: all seasons except the last `test_seasons`.
    Test:  the most recent `test_seasons`.
    """
    max_year = df["year"].max()
    cutoff   = max_year - test_seasons
    train    = df[df["year"] <= cutoff].copy()
    test     = df[df["year"] >  cutoff].copy()
    return train, test


def evaluate(model, scaler, X, y, label=""):
    X_scaled = scaler.transform(X)
    preds    = model.predict(X_scaled)
    mse      = mean_squared_error(y, preds)
    mae      = mean_absolute_error(y, preds)
    rmse     = np.sqrt(mse)

    # Podium accuracy: did we correctly rank the top-3 finishers?
    results_df = X.copy()
    results_df["actual"]    = y.values
    results_df["predicted"] = preds

    print(f"\n📊 {label} Metrics:")
    print(f"   MAE:  {mae:.3f} pts")
    print(f"   RMSE: {rmse:.3f} pts")
    return {"mae": mae, "rmse": rmse, "mse": mse}


def train_model(
    input_csv  = "data/processed/features.csv",
    model_out  = "artifacts/race_predictor.pkl",
    scaler_out = "artifacts/scaler.pkl",
    metrics_out= "metrics/train_metrics.json",
    test_seasons=1,
):
    print("📂 Loading features...")
    df = pd.read_csv(input_csv)

    # Drop rows missing key columns
    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL])

    print(f"   {len(df):,} rows | {df['year'].nunique()} seasons "
          f"({df['year'].min()}–{df['year'].max()})")

    if len(df) < 50:
        print("⚠️  Too few samples to train. Run build_features.py first.")
        return

    # Time-based split
    train, test = time_based_split(df, test_seasons=test_seasons)
    print(f"   Train: {len(train):,} rows ({train['year'].min()}–{train['year'].max()})")
    print(f"   Test:  {len(test):,}  rows ({test['year'].min()}–{test['year'].max()})")

    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train[TARGET_COL]
    X_test  = test[FEATURE_COLS].fillna(0)
    y_test  = test[TARGET_COL]

    # Scale
    scaler   = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Train
    print("\n🏋️  Training GradientBoostingRegressor...")
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_metrics = evaluate(model, scaler, X_train, y_train, label="Train")
    test_metrics  = evaluate(model, scaler, X_test,  y_test,  label="Test ")

    # Feature importance
    print("\n🔍 Feature importances:")
    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for name, imp in importances:
        bar = "█" * int(imp * 40)
        print(f"   {name:<30} {imp:.3f}  {bar}")

    # Save
    os.makedirs(os.path.dirname(model_out),   exist_ok=True)
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)

    joblib.dump(model,  model_out)
    joblib.dump(scaler, scaler_out)
    print(f"\n📂 Model  saved → {model_out}")
    print(f"📂 Scaler saved → {scaler_out}")

    # Save metrics JSON
    metrics = {
        "trained_at":  datetime.now().isoformat(),
        "train_rows":  len(train),
        "test_rows":   len(test),
        "train_years": f"{train['year'].min()}–{train['year'].max()}",
        "test_years":  f"{test['year'].min()}–{test['year'].max()}",
        "features":    FEATURE_COLS,
        "train":       train_metrics,
        "test":        test_metrics,
        "feature_importances": {name: round(float(imp), 4) for name, imp in importances},
    }
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"📂 Metrics saved → {metrics_out}")


if __name__ == "__main__":
    train_model()
