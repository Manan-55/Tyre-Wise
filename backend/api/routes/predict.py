"""
Tyre Wise — Predict Routes
routes/predict.py | /predict and /strategy endpoints
"""

from fastapi import APIRouter, HTTPException
from api.schemas import PredictRequest, PredictResponse, StrategyResponse
import xgboost as xgb
import numpy as np
import os

router = APIRouter()

# ── Load models once at startup ──
MODELS_DIR = "models"
def load_model(compound: str) -> xgb.XGBRegressor:
    path = os.path.join(MODELS_DIR, f"xgb_{compound.lower()}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Model for {compound} not found")
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model

# Pre-load all 3 models
models = {}
for compound in ["soft", "medium", "hard"]:
    try:
        models[compound.upper()] = load_model(compound)
        print(f"✓ Loaded model: {compound.upper()}")
    except Exception as e:
        print(f"✗ Could not load {compound}: {e}")


def build_features(req: PredictRequest) -> np.ndarray:
    """Build feature vector matching train.py feature order."""
    return np.array([[
        req.lap_number,
        req.tyre_life,
        req.tyre_life ** 2,        # tyre_life_squared
        req.is_fresh_tyre,
        req.stint_progress,
        req.deg_rate
    ]])


# ── POST /api/predict ──
@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    compound = req.compound.upper()
    if compound not in models:
        raise HTTPException(status_code=400, detail=f"No model available for {compound}")

    features = build_features(req)
    predicted = float(models[compound].predict(features)[0])

    # Pit recommendation — if tyre life > 25 laps or deg_rate high
    pit_recommended = req.tyre_life > 25 or req.deg_rate > 0.15
    message = "Pit window open — consider stopping" if pit_recommended else "Tyre performance acceptable"

    return PredictResponse(
        compound=compound,
        predicted_lap_time_s=round(predicted, 3),
        tyre_life=req.tyre_life,
        pit_recommended=pit_recommended,
        message=message
    )


# ── GET /api/strategy ──
@router.get("/strategy", response_model=StrategyResponse)
def strategy(
    compound: str = "SOFT",
    current_tyre_life: int = 1,
    lap_number: int = 1,
    deg_rate: float = 0.06
):
    compound = compound.upper()
    if compound not in models:
        raise HTTPException(status_code=400, detail=f"No model for {compound}")

    projected = []
    recommended_pit = lap_number

    for i in range(20):  # project next 20 laps
        tyre_life = current_tyre_life + i
        stint_progress = min(tyre_life / 40.0, 1.0)
        features = np.array([[
            lap_number + i,
            tyre_life,
            tyre_life ** 2,
            0,
            stint_progress,
            deg_rate
        ]])
        lap_time = float(models[compound].predict(features)[0])
        projected.append(round(lap_time, 3))

        # Recommend pit when lap time degrades > 1.5s from first projection
        if i > 3 and lap_time - projected[0] > 1.5 and recommended_pit == lap_number:
            recommended_pit = lap_number + i

    return StrategyResponse(
        compound=compound,
        current_lap_time_s=projected[0],
        projected_lap_times=projected,
        recommended_pit_lap=recommended_pit,
        message=f"Recommended pit window: Lap {recommended_pit}"
    )