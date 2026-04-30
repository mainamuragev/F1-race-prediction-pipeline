import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_model(input_csv="data/processed/features_multiseason.csv",
                model_out="artifacts/f1_predictor.pkl"):
    df = pd.read_csv(input_csv)

    features = ["driver_form_last3", "constructor_form_last3", "grid_advantage"]

    # Convert Points to numeric, drop rows without valid values
    df["Points"] = pd.to_numeric(df["Points"], errors="coerce")
    df = df.dropna(subset=["Points"])

    if df.empty:
        print("⚠️ No valid samples with Points found. Cannot train model.")
        return

    X = df[features].fillna(0)
    y = df["Points"]

    if len(df) < 10:
        print(f"⚠️ Only {len(df)} samples available. Too few to split/train.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"✅ Model trained. Test MSE: {mse:.3f}")

    joblib.dump(model, model_out)
    print(f"📂 Model saved to {model_out}")

if __name__ == "__main__":
    train_model()
