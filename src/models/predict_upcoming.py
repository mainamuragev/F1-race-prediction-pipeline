import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("artifacts/race_predictor.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

# Load upcoming race features
features = pd.read_csv("data/processed/upcoming_features.csv")
scaled = scaler.transform(features.drop(columns=["driverId"]))

# Predict points
features["predicted_points"] = model.predict(scaled)

# Rank drivers by predicted points
ranked = features.sort_values("predicted_points", ascending=False)
print("Predicted upcoming race results:")
print(ranked[["driverId","predicted_points"]].head(10))
