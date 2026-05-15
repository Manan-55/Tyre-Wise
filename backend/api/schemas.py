"""
Tyre Wise — API Schemas
schemas.py | Request & Response Models
"""

from pydantic import BaseModel, Field
from typing import Literal


class PredictRequest(BaseModel):
    compound: Literal["SOFT", "MEDIUM", "HARD"]
    lap_number: int = Field(..., ge=1, le=70, description="Current lap number")
    tyre_life: int = Field(..., ge=1, le=60, description="Laps on current tyre set")
    is_fresh_tyre: int = Field(..., ge=0, le=1, description="1 if new tyre, 0 if used")
    stint_progress: float = Field(..., ge=0.0, le=1.0, description="0.0 to 1.0")
    deg_rate: float = Field(..., description="Degradation rate from preprocess")

    class Config:
        json_schema_extra = {
            "example": {
                "compound": "SOFT",
                "lap_number": 15,
                "tyre_life": 10,
                "is_fresh_tyre": 0,
                "stint_progress": 0.45,
                "deg_rate": 0.06
            }
        }


class PredictResponse(BaseModel):
    compound: str
    predicted_lap_time_s: float
    tyre_life: int
    pit_recommended: bool
    message: str


class StrategyResponse(BaseModel):
    compound: str
    current_lap_time_s: float
    projected_lap_times: list[float]
    recommended_pit_lap: int
    message: str