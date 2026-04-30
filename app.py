"""
app.py
------
FastAPI serving layer for the F1 race prediction model.
Endpoints:
  GET  /health          — liveness check
  POST /predict         — predict points for a single driver's feature vector
  GET  /predict/race    — predict & rank all drivers for the upcoming race
  GET  /logs            — query prediction logs with filters
"""

from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib
import pandas as pd
import datetime
import os
import json
import time

app = FastAPI(
    title="F1 Race Prediction API",
    description="Predicts Formula 1 race points from driver and constructor form features.",
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------

MODEL_PATH  = "artifacts/race_predictor.pkl"
SCALER_PATH = "artifacts/scaler.pkl"

model  = None
scaler = None

def load_artifacts():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Model and scaler loaded.")
    else:
        print(f"⚠️  Model or scaler not found. Run: python main.py --step train")

load_artifacts()

os.makedirs("logs", exist_ok=True)

# ---------------------------------------------------------------------------
# Feature schema — must match FEATURE_COLS in train_model.py exactly
# ---------------------------------------------------------------------------

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


class RaceFeatures(BaseModel):
    driver_form_last5:      float = 0.0
    constructor_form_last5: float = 0.0
    grid_form_last5:        float = 0.0
    pit_form_last5:         float = 0.0
    lap_consistency_last5:  float = 0.0
    circuit_avg_points:     float = 0.0
    cum_season_points:      float = 0.0
    grid:                   float = 10.0
    driver_id:              str   = "unknown"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "scaler_loaded":scaler is not None,
        "timestamp":    datetime.datetime.now().isoformat(),
    }


@app.post("/predict")
def predict(features: RaceFeatures):
    if model is None or scaler is None:
        return {"error": "Model not loaded. Run python main.py --step train first."}

    start = time.time()

    data       = pd.DataFrame([{col: getattr(features, col) for col in FEATURE_COLS}])
    data_scaled = scaler.transform(data)
    prediction  = float(model.predict(data_scaled)[0])
    duration_ms = round((time.time() - start) * 1000, 2)

    log_entry = {
        "timestamp":       datetime.datetime.now().isoformat(),
        "driver_id":       features.driver_id,
        "features":        {col: getattr(features, col) for col in FEATURE_COLS},
        "prediction":      prediction,
        "response_time_ms":duration_ms,
    }
    with open("logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "driver_id":        features.driver_id,
        "predicted_points": prediction,
        "features_used":    {col: getattr(features, col) for col in FEATURE_COLS},
        "response_time_ms": duration_ms,
    }


@app.get("/predict/race")
def predict_race():
    """Predict and rank all drivers for the upcoming race."""
    if model is None or scaler is None:
        return {"error": "Model not loaded. Run python main.py --step train first."}

    upcoming_path = "data/processed/upcoming_features.csv"
    if not os.path.exists(upcoming_path):
        return {"error": "Upcoming features not found. Run python main.py --step upcoming first."}

    df = pd.read_csv(upcoming_path)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    X_scaled = scaler.transform(df[FEATURE_COLS])
    df["predicted_points"] = model.predict(X_scaled)

    ranked = df.sort_values("predicted_points", ascending=False).reset_index(drop=True)
    ranked.index += 1

    driver_col = "driverRef" if "driverRef" in ranked.columns else "driverId"
    const_col  = "constructorRef" if "constructorRef" in ranked.columns else "constructorId"

    results = []
    for pos, row in ranked.iterrows():
        results.append({
            "position":        pos,
            "driver":          row.get(driver_col, "unknown"),
            "constructor":     row.get(const_col, "unknown"),
            "predicted_points":round(max(0, row["predicted_points"]), 2),
        })

    # Next race info
    race_info = {}
    if os.path.exists("data/processed/next_race_info.json"):
        with open("data/processed/next_race_info.json") as f:
            race_info = json.load(f)

    return {
        "race":        race_info,
        "predictions": results,
        "generated_at":datetime.datetime.now().isoformat(),
    }


@app.get("/logs")
def get_logs(
    limit:            int   = Query(10,  ge=1, le=500),
    min_response_ms:  float = Query(None),
    max_response_ms:  float = Query(None),
    driver_id:        str   = Query(None),
    start:            str   = Query(None, description="ISO datetime"),
    end:              str   = Query(None, description="ISO datetime"),
    recent:           str   = Query(None, description="hour | day | week"),
    order:            str   = Query("desc", regex="^(asc|desc)$"),
):
    log_path = "logs/predictions.jsonl"
    if not os.path.exists(log_path):
        return {"recent_predictions": []}

    with open(log_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    # Filters
    if driver_id:
        entries = [e for e in entries if e.get("driver_id") == driver_id]
    if min_response_ms is not None:
        entries = [e for e in entries if e.get("response_time_ms", 0) >= min_response_ms]
    if max_response_ms is not None:
        entries = [e for e in entries if e.get("response_time_ms", 0) <= max_response_ms]

    def parse_dt(s):
        return datetime.datetime.fromisoformat(s)

    if start:
        start_dt = parse_dt(start)
        entries  = [e for e in entries if parse_dt(e["timestamp"]) >= start_dt]
    if end:
        end_dt  = parse_dt(end)
        entries = [e for e in entries if parse_dt(e["timestamp"]) <= end_dt]
    if recent:
        now    = datetime.datetime.now()
        deltas = {"hour": datetime.timedelta(hours=1), "day": datetime.timedelta(days=1), "week": datetime.timedelta(weeks=1)}
        if recent in deltas:
            cutoff  = now - deltas[recent]
            entries = [e for e in entries if parse_dt(e["timestamp"]) >= cutoff]

    entries.sort(key=lambda e: parse_dt(e["timestamp"]), reverse=(order == "desc"))
    return {"recent_predictions": entries[:limit], "total_returned": min(limit, len(entries))}
