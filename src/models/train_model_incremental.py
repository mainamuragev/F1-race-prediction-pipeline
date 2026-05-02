import pandas as pd
import joblib
import os
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    'driver_form_last5', 'grid', 'constructor_form_last5', 'pit_form_last5',
    'cum_season_points', 'lap_consistency_last5', 'circuit_avg_points', 'grid_form_last5'
]

def train_incremental():
    df = pd.read_csv('data/processed/features.csv')
    if df.empty:
        print("❌ No features found. Skipping training.")
        # Create dummy model and scaler to avoid later errors
        os.makedirs('artifacts', exist_ok=True)
        dummy_model = SGDRegressor()
        joblib.dump(dummy_model, 'artifacts/race_predictor_sgd.pkl')
        dummy_scaler = StandardScaler()
        joblib.dump(dummy_scaler, 'artifacts/scaler_sgd.pkl')
        return
    X = df[FEATURE_COLS].fillna(0)
    y = df['position']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = SGDRegressor(loss='squared_error', learning_rate='adaptive', eta0=0.01, max_iter=1000, tol=1e-3)
    model.fit(X_scaled, y)  # use fit, not partial_fit, for initial training
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(model, 'artifacts/race_predictor_sgd.pkl')
    joblib.dump(scaler, 'artifacts/scaler_sgd.pkl')
    print("✅ Incremental model training complete (SGD).")

def update_model(new_race_df):
    """Call after each race with new data (single race)."""
    model = joblib.load('artifacts/race_predictor_sgd.pkl')
    scaler = joblib.load('artifacts/scaler_sgd.pkl')
    X_new = new_race_df[FEATURE_COLS].fillna(0)
    X_new_scaled = scaler.transform(X_new)
    y_new = new_race_df['position']
    model.partial_fit(X_new_scaled, y_new)
    joblib.dump(model, 'artifacts/race_predictor_sgd.pkl')
    print("✅ Model updated with new race.")

if __name__ == "__main__":
    train_incremental()
