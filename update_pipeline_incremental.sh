#!/bin/bash
cd ~/F1-race-prediction-pipeline
source .venv/bin/activate
echo "=== FETCHING LATEST RACE (with practice/weather) ==="
python src/ingestion/fetch_data.py
echo "=== BUILDING FULL FEATURES ==="
python src/features/build_features.py
echo "=== UPDATING MODEL INCREMENTALLY ==="
python src/models/train_model_incremental.py
echo "=== FETCHING NEXT RACE GRID (if qualifying done) ==="
python src/ingestion/fetch_upcoming_qualifying.py || echo "Qualifying not yet available, skipping."
echo "=== BUILDING UPCOMING FEATURES ==="
python src/features/build_upcoming_features.py
echo "=== PREDICTING NEXT RACE ==="
python src/models/predict_upcoming.py
echo "=== DONE ==="
