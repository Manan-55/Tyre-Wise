"""
PITWALL — Live Data Service
live_fetch.py | OpenF1 API Integration
Runs as a FastAPI background task, fetches live lap + stint data every 30s
Author: Manan Prajapati
"""

import os
import httpx
import asyncio
import logging
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logger = logging.getLogger("pitwall.live")

# ─────────────────────────────────────────
# OPENF1 CONFIG
# ─────────────────────────────────────────

OPENF1_BASE = "https://api.openf1.org/v1"
FETCH_INTERVAL = 30  # seconds between fetches during live race

# ─────────────────────────────────────────
# IN-MEMORY STORE
# ─────────────────────────────────────────

live_state = {
    "session_key": None,
    "meeting_name": None,
    "last_updated": None,
    "is_live": False,
    "drivers": {}
}


# ─────────────────────────────────────────
# OPENF1 FETCH HELPERS
# ─────────────────────────────────────────

async def get_latest_session() -> Optional[dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{OPENF1_BASE}/sessions?session_type=Race&session_key=latest")
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data[-1] if data else None


async def get_latest_laps(session_key: int) -> list:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{OPENF1_BASE}/laps?session_key={session_key}", timeout=15)
        if resp.status_code != 200:
            return []
        return resp.json()


async def get_stints(session_key: int) -> list:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{OPENF1_BASE}/stints?session_key={session_key}", timeout=15)
        if resp.status_code != 200:
            return []
        return resp.json()


async def get_drivers(session_key: int) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{OPENF1_BASE}/drivers?session_key={session_key}", timeout=15)
        if resp.status_code != 200:
            return {}
        drivers = resp.json()
        return {
            d["driver_number"]: {
                "name": d.get("full_name", "Unknown"),
                "abbreviation": d.get("name_acronym", "???"),
                "team": d.get("team_name", "Unknown")
            }
            for d in drivers
        }


# ─────────────────────────────────────────
# DATA PROCESSING
# ─────────────────────────────────────────

def build_driver_state(laps: list, stints: list, drivers: dict) -> dict:
    driver_laps = {}
    for lap in laps:
        dn = lap.get("driver_number")
        if not dn or lap.get("is_pit_out_lap"):
            continue
        if lap.get("lap_duration") is None:
            continue
        if dn not in driver_laps:
            driver_laps[dn] = []
        driver_laps[dn].append(lap)

    driver_stints = {}
    for stint in stints:
        dn = stint.get("driver_number")
        if not dn:
            continue
        if dn not in driver_stints:
            driver_stints[dn] = []
        driver_stints[dn].append(stint)

    result = {}

    for dn, d_laps in driver_laps.items():
        d_laps.sort(key=lambda x: x.get("lap_number", 0))
        latest_lap = d_laps[-1]

        compound = "UNKNOWN"
        tyre_life = 1
        stint_number = 1

        if dn in driver_stints:
            d_stints = sorted(driver_stints[dn], key=lambda x: x.get("stint_number", 0))
            latest_stint = d_stints[-1]
            compound = latest_stint.get("compound", "UNKNOWN").upper()
            tyre_life = latest_stint.get("tyre_age_at_start", 0) + (
                latest_lap.get("lap_number", 1) - latest_stint.get("lap_start", 1)
            )
            tyre_life = max(1, tyre_life)
            stint_number = latest_stint.get("stint_number", 1)

        deg_rate = 0.06
        if len(d_laps) >= 3:
            recent = d_laps[-3:]
            times = [l.get("lap_duration", 0) for l in recent if l.get("lap_duration")]
            if len(times) >= 2:
                deg_rate = round((times[-1] - times[0]) / len(times), 4)

        lap_number = latest_lap.get("lap_number", 1)
        stint_progress = round(min(tyre_life / 40.0, 1.0), 4)
        driver_info = drivers.get(dn, {"name": f"Driver {dn}", "abbreviation": str(dn), "team": "Unknown"})

        result[dn] = {
            "driver_number": dn,
            "abbreviation": driver_info["abbreviation"],
            "name": driver_info["name"],
            "team": driver_info["team"],
            "lap_number": lap_number,
            "compound": compound,
            "tyre_life": tyre_life,
            "stint_number": stint_number,
            "last_lap_time_s": latest_lap.get("lap_duration"),
            "deg_rate": deg_rate,
            "stint_progress": stint_progress,
            "is_fresh_tyre": 1 if tyre_life <= 2 else 0,
            "predicted_lap_time_s": None,
            "pit_recommended": False,
            "pit_message": ""
        }

    return result


def run_predictions(driver_state: dict, models: dict) -> dict:
    import numpy as np
    for dn, state in driver_state.items():
        compound = state["compound"]
        if compound not in models:
            continue
        features = np.array([[
            state["lap_number"],
            state["tyre_life"],
            state["tyre_life"] ** 2,
            state["is_fresh_tyre"],
            state["stint_progress"],
            state["deg_rate"]
        ]])
        predicted = float(models[compound].predict(features)[0])
        pit_recommended = state["tyre_life"] > 25 or state["deg_rate"] > 0.15
        state["predicted_lap_time_s"] = round(predicted, 3)
        state["pit_recommended"] = pit_recommended
        state["pit_message"] = "🔴 Pit window open" if pit_recommended else "🟢 Stay out"
    return driver_state


# ─────────────────────────────────────────
# SUPABASE SAVE
# ─────────────────────────────────────────

def save_to_supabase(session_key: int, meeting_name: str, driver_state: dict):
    try:
        # Upsert session
        supabase.table("sessions").upsert({
    "session_key": session_key,
    "meeting_name": meeting_name,
    "total_laps": max([d["lap_number"] for d in driver_state.values()], default=0)
}, on_conflict="session_key").execute()

        # Get session id
        session = supabase.table("sessions").select("id").eq("session_key", session_key).execute()
        session_id = session.data[0]["id"]

        # Insert predictions
        rows = []
        for state in driver_state.values():
            rows.append({
                "session_id": session_id,
                "driver_number": state["driver_number"],
                "driver_name": state["name"],
                "abbreviation": state["abbreviation"],
                "team": state["team"],
                "lap_number": state["lap_number"],
                "compound": state["compound"],
                "tyre_life": state["tyre_life"],
                "stint_number": state["stint_number"],
                "deg_rate": state["deg_rate"],
                "stint_progress": state["stint_progress"],
                "predicted_lap_time_s": state["predicted_lap_time_s"],
                "pit_recommended": state["pit_recommended"]
            })

        if rows:
            supabase.table("predictions").insert(rows).execute()
            logger.info(f"  ✓ Saved {len(rows)} predictions to Supabase")

    except Exception as e:
        logger.error(f"Supabase save error: {e}")


# ─────────────────────────────────────────
# BACKGROUND LOOP
# ─────────────────────────────────────────

async def live_fetch_loop(models: dict):
    logger.info("🏎  PITWALL live fetch service started")

    while True:
        try:
            session = await get_latest_session()

            if not session:
                live_state["is_live"] = False
                await asyncio.sleep(60)
                continue

            session_key = session.get("session_key")
            meeting_name = session.get("meeting_name", "Unknown GP")

            live_state["session_key"] = session_key
            live_state["meeting_name"] = meeting_name
            live_state["is_live"] = True

            logger.info(f"Fetching live data: {meeting_name} (session {session_key})")

            laps, stints, drivers = await asyncio.gather(
                get_latest_laps(session_key),
                get_stints(session_key),
                get_drivers(session_key)
            )

            logger.info(f"  Laps: {len(laps)} | Stints: {len(stints)} | Drivers: {len(drivers)}")

            driver_state = build_driver_state(laps, stints, drivers)
            driver_state = run_predictions(driver_state, models)

            # Save to Supabase
            save_to_supabase(session_key, meeting_name, driver_state)

            live_state["drivers"] = driver_state
            live_state["last_updated"] = datetime.utcnow().isoformat()

            logger.info(f"  ✓ Updated {len(driver_state)} drivers")

        except Exception as e:
            logger.error(f"Live fetch error: {e}")

        await asyncio.sleep(FETCH_INTERVAL)