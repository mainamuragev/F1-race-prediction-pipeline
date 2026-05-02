#!/bin/bash
cd ~/F1-race-prediction-pipeline
source .venv/bin/activate
echo "=== Fetching latest race ==="
python src/ingestion/fetch_data.py
echo "=== Building features ==="
python src/features/build_features.py
echo "=== Building upcoming features ==="
python src/features/build_upcoming_features.py
echo "=== Training model ==="
python src/models/train_model.py
echo "=== Predicting next race ==="
python src/models/predict_upcoming.py
echo "=== Done ==="
