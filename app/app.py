from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load trained model
model = joblib.load("artifacts/race_predictor.pkl")

app = FastAPI()

# Define input schema
class RaceFeatures(BaseModel):
    driver_form_last5: float
    constructor_form_last5: float
    grid_form_last5: float
    pit_form_last5: float
    lap_consistency_last5: float

@app.post("/predict")
def predict(features: RaceFeatures):
    # Convert input to DataFrame
    data = pd.DataFrame([features.dict()])

    # Make prediction
    prediction = model.predict(data)[0]

    return {"predicted_points": prediction}
