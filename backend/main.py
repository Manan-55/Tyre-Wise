"""
PITWALL — FastAPI Backend
main.py | Entry Point
Author: Manan Prajapati
"""

from database import init_db
import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import predict
from live_fetch import live_fetch_loop, live_state
from api.routes.predict import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(
    title="PITWALL API",
    description="F1 Tyre Degradation Predictor — XGBoost per compound models + OpenF1 live data",
    version="1.0.0"
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ──
app.include_router(predict.router, prefix="/api")


# ── Start live fetch background loop on startup ──
@app.on_event("startup")
async def startup_event():
    #await init_db()
    #logging.info("✓ Database tables created")
    asyncio.create_task(live_fetch_loop(models))
    logging.info("✓ Live fetch background service started")


# ── Health check ──
@app.get("/")
def root():
    return {"status": "PITWALL API is running 🏎"}


# ── Live data endpoint — frontend hits this ──
@app.get("/api/live")
def get_live_data():
    return {
        "session": live_state["meeting_name"],
        "session_key": live_state["session_key"],
        "is_live": live_state["is_live"],
        "last_updated": live_state["last_updated"],
        "drivers": list(live_state["drivers"].values())
    }