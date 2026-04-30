from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import datetime
import os
import json
import time

app = FastAPI()

# Load trained model and scaler
model = joblib.load("artifacts/race_predictor.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

class RaceFeatures(BaseModel):
    driver_form_last5: float = 0.0
    constructor_form_last5: float = 0.0
    grid_form_last5: float = 0.0
    pit_form_last5: float = 0.0
    lap_consistency_last5: float = 0.0

@app.get("/health")
def health():
    return {"status": "ok", "message": "API is running"}

@app.post("/predict")
def predict(features: RaceFeatures):
    start_time = time.time()

    # Prepare data
    data = pd.DataFrame([features.dict()])
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)[0]
    duration = time.time() - start_time

    # Log as JSON line
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "features": features.dict(),
        "prediction": float(prediction),
        "response_time_ms": round(duration * 1000, 2)
    }
    with open("logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "predicted_points": float(prediction),
        "features_used": features.dict(),
        "response_time_ms": round(duration * 1000, 2)
    }

@app.get("/logs")
def get_logs(limit: int = 10,
             min_response_time: float = None,
             max_response_time: float = None,
             grid_form_max: float = None,
             driver_form_min: float = None,
             start: str = None,
             end: str = None,
             recent: str = None,
             order: str = "desc"):  # default newest-first
    try:
        with open("logs/predictions.jsonl", "r") as f:
            lines = f.readlines()
        structured = [json.loads(line) for line in lines]

        # Apply filters safely
        if min_response_time is not None:
            structured = [entry for entry in structured
                          if "response_time_ms" in entry and entry["response_time_ms"] >= min_response_time]
        if max_response_time is not None:
            structured = [entry for entry in structured
                          if "response_time_ms" in entry and entry["response_time_ms"] <= max_response_time]
        if grid_form_max is not None:
            structured = [entry for entry in structured
                          if "features" in entry and entry["features"].get("grid_form_last5") is not None
                          and entry["features"]["grid_form_last5"] <= grid_form_max]
        if driver_form_min is not None:
            structured = [entry for entry in structured
                          if "features" in entry and entry["features"].get("driver_form_last5") is not None
                          and entry["features"]["driver_form_last5"] >= driver_form_min]

        # Timestamp filters
        if start is not None:
            start_dt = datetime.datetime.fromisoformat(start)
            structured = [entry for entry in structured
                          if datetime.datetime.fromisoformat(entry["timestamp"]) >= start_dt]
        if end is not None:
            end_dt = datetime.datetime.fromisoformat(end)
            structured = [entry for entry in structured
                          if datetime.datetime.fromisoformat(entry["timestamp"]) <= end_dt]

        # Recent shortcut
        if recent is not None:
            now = datetime.datetime.now()
            if recent == "hour":
                cutoff = now - datetime.timedelta(hours=1)
            elif recent == "day":
                cutoff = now - datetime.timedelta(days=1)
            elif recent == "week":
                cutoff = now - datetime.timedelta(weeks=1)
            else:
                cutoff = None
            if cutoff:
                structured = [entry for entry in structured
                              if datetime.datetime.fromisoformat(entry["timestamp"]) >= cutoff]

        # Sorting
        structured.sort(key=lambda e: datetime.datetime.fromisoformat(e["timestamp"]),
                        reverse=(order == "desc"))

        # Limit after sorting
        structured = structured[:limit]

        return {"recent_predictions": structured}
    except FileNotFoundError:
        return {"recent_predictions": []}
