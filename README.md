```
███████╗ ██╗     ██████╗   █████╗   ██████╗███████╗    ██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗ ██████╗ ██████╗
██╔════╝ ██║     ██╔══██╗ ██╔══██╗ ██╔════╝██╔════╝    ██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
█████╗   ██║     ██████╔╝ ███████║ ██║     █████╗      ██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║   ██║   ██║██████╔╝
██╔══╝   ██║     ██╔══██╗ ██╔══██║ ██║     ██╔══╝      ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║   ██║   ██║██╔══██╗
██║      ███████╗██║  ██║ ██║  ██║ ╚██████╗███████╗    ██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║   ╚██████╔╝██║  ██║
╚═╝      ╚══════╝╚═╝  ╚═╝ ╚═╝  ╚═╝  ╚═════╝╚══════╝    ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
```

<div align="center">

🏎️ **An intelligent Formula 1 race outcome predictor** 🏎️

*Leverages historical performance, driver form, and constructor strength to forecast race results*

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)

</div>

---

## 🗺️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        main.py  (orchestrator)                      │
└──────────┬──────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐     ┌──────────────────────────────────────┐
│   STEP 1: CLEAN      │     │  Fixes                               │
│                      │────▶│  🧹 Ergast \N null handling          │
│  clean_data.py       │     │  🕐 Lap time string → milliseconds   │
│                      │     │  📊 Type coercion + DNF filling      │
└──────────┬───────────┘     └──────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐     ┌──────────────────────────────────────┐
│  STEP 2: FEATURES    │     │  8 Features Engineered               │
│                      │────▶│  📈 driver_form_last5                │
│  build_features.py   │     │  🏗️  constructor_form_last5          │
│                      │     │  🚦 grid_form_last5                  │
│                      │     │  ⏱️  pit_form_last5                   │
│                      │     │  🔄 lap_consistency_last5            │
│                      │     │  🏟️  circuit_avg_points               │
│                      │     │  🏆 cum_season_points                │
│                      │     │  🔢 grid (qualifying position)       │
└──────────┬───────────┘     └──────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐     ┌──────────────────────────────────────┐
│   STEP 3: TRAIN      │     │  Model Details                       │
│                      │────▶│  🤖 GradientBoostingRegressor        │
│  train_model.py      │     │  📅 Time-based train/test split      │
│                      │     │  💾 Saves model + scaler             │
│                      │     │  📊 MAE + RMSE + feature importance  │
└──────────┬───────────┘     └──────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐     ┌──────────────────────────────────────┐
│  STEP 4: UPCOMING    │     │  For Next Race                       │
│                      │────▶│  🗓️  Fetch next race from Ergast API │
│  build_upcoming_     │     │  📐 Compute per-driver feature rows  │
│  features.py         │     │  💾 upcoming_features.csv            │
└──────────┬───────────┘     └──────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐     ┌──────────────────────────────────────┐
│   STEP 5: PREDICT    │     │  Output                              │
│                      │────▶│  🏁 Ranked driver leaderboard        │
│  predict_upcoming.py │     │  💾 predictions.csv                  │
│                      │     │  🌐 FastAPI /predict/race endpoint   │
└──────────────────────┘     └──────────────────────────────────────┘
```

---

## 📁 Project Structure

```
f1-race-prediction-pipeline/
│
├── 🚀 main.py                           # Pipeline orchestrator (start here)
├── 🌐 app.py                            # FastAPI serving layer
│
├── src/
│   ├── ingestion/
│   │   └── fetch_data.py                # Local Kaggle dataset loader
│   │
│   ├── features/
│   │   ├── ⭐ build_features.py         # Unified feature engineering
│   │   ├── build_upcoming_features.py   # Features for next race
│   │   ├── ingest_hybrid.py             # FastF1 + Kaggle hybrid
│   │   ├── ingest_ergast.py             # Ergast API ingestion
│   │   ├── ingest_fastf1.py             # FastF1 telemetry
│   │   └── ingest_multiseason.py        # Multi-season backfill
│   │
│   ├── models/
│   │   ├── train_model.py               # Training + evaluation
│   │   └── predict_upcoming.py          # Predict & rank
│   │
│   └── processing/
│       └── clean_data.py                # Raw data cleaning
│
├── data/
│   ├── raw/                             # Kaggle Ergast CSVs
│   └── processed/                       # Generated features + predictions
│
├── artifacts/                           # Saved model + scaler (.pkl)
├── logs/                                # Prediction logs (.jsonl)
└── metrics/                             # Training metrics (.json)
```

---

## ⚡ Quickstart

### 1. Install dependencies

```bash
uv sync
```

### 2. Add raw data

Place Kaggle Ergast CSVs into `data/raw/`:

```
data/raw/
├── results.csv              ← race results per driver
├── races.csv                ← race calendar metadata
├── drivers.csv              ← driver info
├── constructors.csv         ← constructor info
├── qualifying.csv           ← qualifying positions
├── pit_stops.csv            ← pit stop times
├── lap_times.csv            ← per-lap times
├── constructor_standings.csv
└── circuits.csv
```

> Download: [Kaggle F1 World Championship Dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)

### 3. Run the full pipeline

```bash
python main.py
```

### 4. Run individual steps

```bash
python main.py --step clean      # Step 1: clean raw CSVs
python main.py --step features   # Step 2: build feature set
python main.py --step train      # Step 3: train model
python main.py --step upcoming   # Step 4: build upcoming race features
python main.py --step predict    # Step 5: predict next race
```

### 5. Start the API

```bash
uvicorn app:app --reload
```

Interactive docs → **http://localhost:8000/docs**

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Liveness check + model load status |
| `POST` | `/predict` | Predict points for a single driver |
| `GET`  | `/predict/race` | Full ranked leaderboard for next race |
| `GET`  | `/logs` | Query prediction logs with filters |

### Predict a single driver

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "driver_id": "max_verstappen",
    "driver_form_last5": 18.5,
    "constructor_form_last5": 35.0,
    "grid_form_last5": 1.8,
    "pit_form_last5": 24500,
    "lap_consistency_last5": 850,
    "circuit_avg_points": 15.0,
    "cum_season_points": 120,
    "grid": 1
  }'
```

### Full race leaderboard

```bash
curl http://localhost:8000/predict/race
```

```json
{
  "race": { "race_name": "Miami Grand Prix", "date": "2026-05-04" },
  "predictions": [
    { "position": 1, "driver": "max_verstappen", "predicted_points": 18.42 },
    { "position": 2, "driver": "norris",         "predicted_points": 15.31 },
    { "position": 3, "driver": "leclerc",        "predicted_points": 12.87 }
  ]
}
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Algorithm | Gradient Boosting Regressor |
| Target variable | Race points scored |
| Train/test split | Time-based — last season held out |
| Number of features | 8 |
| Evaluation metrics | MAE + RMSE |

---

## 📊 Data Sources

| Source | Coverage | Used For |
|--------|----------|----------|
| [Kaggle Ergast Dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) | 1950–2024 | Historical training data |
| [FastF1](https://github.com/theOehrly/Fast-F1) | 2018–present | Telemetry, lap times, pit stops |
| [Ergast API](http://ergast.com/mrd/) | Current season | Live race results |

---

<div align="center">

Built with ❤️ and way too much coffee ☕

*"To finish first, first you must finish"* — Stirling Moss

</div>
