import joblib
import pandas as pd
import numpy as np

FEATURE_COLS = [
    'driver_form_last5', 'grid', 'constructor_form_last5', 'pit_form_last5',
    'cum_season_points', 'lap_consistency_last5', 'circuit_avg_points', 'grid_form_last5'
]

def predict_upcoming():
    model = joblib.load('artifacts/race_predictor_sgd.pkl')
    scaler = joblib.load('artifacts/scaler_sgd.pkl')
    upcoming = pd.read_csv('data/processed/upcoming_features.csv')
    for col in FEATURE_COLS:
        if col not in upcoming.columns:
            upcoming[col] = 0
    X = upcoming[FEATURE_COLS].fillna(0)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    upcoming['predicted_position'] = preds
    sorted_df = upcoming.sort_values('predicted_position')
    print("\n🏁 UPDATED PREDICTION (with qualifying, practice, weather) 🏁\n")
    print(f"{'Pos':<4} {'Driver':<30} {'Team':<20} {'Pred Finish':<12}")
    for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
        pred = max(1, min(20, round(row['predicted_position'], 1)))
        print(f"{i:<4} {row['driver']:<30} {row['team']:<20} {pred:<12}")
    sorted_df.to_csv('data/processed/predictions.csv', index=False)

if __name__ == "__main__":
    predict_upcoming()
