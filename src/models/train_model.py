import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def main():
    df = pd.read_csv("data/processed/features.csv")

    features = [
        "driver_form_last5",
        "constructor_form_last5",
        "grid_form_last5",
        "pit_form_last5",
        "lap_consistency_last5"
    ]

    X = df[features].fillna(0)
    y = df["Points"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    print(f"✅ Model trained with scaling. RMSE: {rmse:.2f}")

    # Save both model and scaler
    joblib.dump(model, "artifacts/race_predictor.pkl")
    joblib.dump(scaler, "artifacts/scaler.pkl")
    print("💾 Model and scaler saved to artifacts/")

if __name__ == "__main__":
    main()
