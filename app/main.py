from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("artifacts/race_predictor.pkl")

@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])
    pred = model.predict(df)
    return {"prediction": int(pred[0])}
