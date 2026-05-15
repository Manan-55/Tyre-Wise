# PITWALL 🏎

> F1 Tyre Degradation Predictor — Real-time pit stop strategy powered by XGBoost + OpenF1

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-GPU-orange?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green?style=for-the-badge&logo=fastapi)
![OpenF1](https://img.shields.io/badge/OpenF1-Live-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## What is PITWALL?

PITWALL is an end-to-end F1 tyre strategy system that predicts lap time degradation per tyre compound and recommends pit stops in real time during live races.

It pulls live telemetry from the OpenF1 API, runs compound-specific XGBoost models, and exposes predictions through a FastAPI backend — updating every 30 seconds during a race.

---

## Architecture

```
FastF1 (Historical)         OpenF1 (Live Race)
       │                           │
  fetch.py                  live_fetch.py
       │                           │
  data/raw/              Background Service
       │                    (every 30s)
 preprocess.py                     │
       │                           ▼
 data/processed/          /api/live endpoint
       │                           │
   train.py                        │
       │                           │
 backend/models/   ────────────────┘
  xgb_soft.json                    │
  xgb_medium.json          FastAPI Backend
  xgb_hard.json                    │
                      ┌────────────┼────────────┐
                      │            │            │
               /api/predict  /api/strategy  /api/live
                      │
                 Next.js Frontend
```

---

## Key Features

- **Per-compound XGBoost models** — separate models for SOFT, MEDIUM, HARD tyres. Each compound degrades differently; one generic model loses that signal.
- **GPU accelerated training** — XGBoost runs on CUDA for faster training
- **Live race integration** — OpenF1 API feeds real driver/compound/tyre data every 30 seconds during a race
- **Pit stop recommender** — flags drivers whose tyres are past optimal window
- **Strategy projection** — projects next 20 laps and recommends pit lap
- **REST API** — FastAPI backend with auto-generated Swagger docs

---

## ML Approach

### Feature Engineering
| Feature | Description |
|---|---|
| `TyreLife` | Laps completed on current tyre set |
| `tyre_life_squared` | Models the non-linear degradation cliff |
| `LapNumber` | Current lap of the race |
| `stint_progress` | How far into the stint (0.0 → 1.0) |
| `is_fresh_tyre` | 1 if new tyre, 0 if used set |
| `deg_rate` | Degradation slope (s/lap) from recent laps |

### Model Performance (MAE)
| Compound | Test MAE |
|---|---|
| SOFT | 0.429s |
| MEDIUM | 0.723s |
| HARD | 0.919s |

Trained on 7,432 clean laps across 8 races (2023–2024 seasons).

---

## API Endpoints

### `POST /api/predict`
Predict lap time for given tyre state.

```json
// Request
{
  "compound": "SOFT",
  "lap_number": 15,
  "tyre_life": 10,
  "is_fresh_tyre": 0,
  "stint_progress": 0.45,
  "deg_rate": 0.06
}

// Response
{
  "compound": "SOFT",
  "predicted_lap_time_s": 97.524,
  "tyre_life": 10,
  "pit_recommended": false,
  "message": "Tyre performance acceptable"
}
```

### `GET /api/strategy`
Project next 20 laps and recommend pit window.

```
GET /api/strategy?compound=SOFT&current_tyre_life=10&lap_number=15&deg_rate=0.06
```

### `GET /api/live`
Live predictions for all drivers during an active race session.

```json
{
  "session": "Miami Grand Prix",
  "is_live": true,
  "last_updated": "2026-05-03T17:22:39Z",
  "drivers": [
    {
      "abbreviation": "VER",
      "name": "Max VERSTAPPEN",
      "team": "Red Bull Racing",
      "compound": "HARD",
      "tyre_life": 50,
      "predicted_lap_time_s": 85.455,
      "pit_recommended": true,
      "pit_message": "🔴 Pit window open"
    }
  ]
}
```

---

## Project Structure

```
PITWALL/
├── fetch.py                      # FastF1 historical data pipeline
├── preprocess.py                 # Feature engineering + cleaning
├── train.py                      # Per-compound XGBoost training
├── .gitignore
├── README.md
├── data/
│   ├── cache/                    # FastF1 cache (gitignored)
│   ├── raw/                      # Per-race CSVs + master_laps.csv
│   └── processed/                # ML-ready master_processed.csv
└── backend/
    ├── main.py                   # FastAPI entry point + live fetch startup
    ├── live_fetch.py             # OpenF1 background service (30s polling)
    ├── models/                   # Trained XGBoost models (gitignored)
    │   ├── xgb_soft.json
    │   ├── xgb_medium.json
    │   ├── xgb_hard.json
    │   └── feature_schema.json
    └── api/
        ├── schemas.py            # Pydantic request/response models
        └── routes/
            └── predict.py        # /predict and /strategy endpoints
```

---

## Getting Started

### Prerequisites
- Python 3.12+
- GPU with CUDA support (optional, falls back to CPU)

### Installation

```bash
git clone https://github.com/Manan-55/Tyre-Wise.git
cd Tyre-Wise
pip install fastf1 pandas numpy xgboost scikit-learn fastapi uvicorn httpx joblib
```

### Run the Pipeline

```bash
# 1. Fetch historical F1 data
python fetch.py

# 2. Preprocess + feature engineering
python preprocess.py

# 3. Train per-compound models
python train.py
```

### Start the API

```bash
cd backend
uvicorn main:app --reload
```

API docs available at `http://localhost:8000/docs`

---

## Data Sources

| Source | Usage |
|---|---|
| [FastF1](https://github.com/theOehrly/Fast-F1) | Historical lap + tyre data (2023–2024) |
| [OpenF1](https://openf1.org) | Live race telemetry during race weekends |

---

## Tech Stack

`Python` · `XGBoost` · `FastAPI` · `FastF1` · `OpenF1` · `pandas` · `scikit-learn` · `httpx` · `uvicorn`

---

## Author

**Manan Prajapati** — [GitHub](https://github.com/Manan-55) · CS Undergrad → Data Science & ML
