"""
predict_upcoming.py
-------------------
Loads the trained model + scaler and predicts points for the upcoming race.
Outputs a ranked leaderboard to stdout and saves to data/processed/predictions.csv.
"""

import pandas as pd
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


def predict_upcoming(
    features_csv = "data/processed/upcoming_features.csv",
    model_path   = "artifacts/race_predictor.pkl",
    scaler_path  = "artifacts/scaler.pkl",
    output_csv   = "data/processed/predictions.csv",
    race_info_json = "data/processed/next_race_info.json",
):
    # --- Load model & scaler ---
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_model.py first."
        )
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Run train_model.py first."
        )

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # --- Load upcoming features ---
    if not os.path.exists(features_csv):
        raise FileNotFoundError(
            f"Upcoming features not found at {features_csv}. "
            "Run build_upcoming_features.py first."
        )

    df = pd.read_csv(features_csv)

    # Fill missing feature columns with 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    X = df[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    # --- Predict ---
    df["predicted_points"] = model.predict(X_scaled)

    # Rank
    ranked = df.sort_values("predicted_points", ascending=False).reset_index(drop=True)
    ranked.index += 1  # 1-based rank

    # --- Print leaderboard ---
    race_label = "Upcoming Race"
    if os.path.exists(race_info_json):
        with open(race_info_json) as f:
            info = json.load(f)
        race_label = f"{info.get('race_name', 'Upcoming Race')}  ({info.get('date', '')})"

    print(f"\n🏎️  Predicted Results — {race_label}")
    print("=" * 60)
    print(f"{'Pos':<5} {'Driver':<25} {'Constructor':<20} {'Pred Pts':>8}")
    print("-" * 60)

    driver_col = "driverRef" if "driverRef" in ranked.columns else "driverId"
    const_col  = "constructorRef" if "constructorRef" in ranked.columns else "constructorId"

    for pos, row in ranked.iterrows():
        pts = max(0, row["predicted_points"])
        print(f"{pos:<5} {str(row[driver_col]):<25} {str(row[const_col]):<20} {pts:>8.2f}")

    # --- Save ---
    os.makedirs("data/processed", exist_ok=True)
    ranked["predicted_at"] = datetime.now().isoformat()
    ranked.to_csv(output_csv, index=False)
    print(f"\n📂 Predictions saved → {output_csv}")

    return ranked


if __name__ == "__main__":
    predict_upcoming()
