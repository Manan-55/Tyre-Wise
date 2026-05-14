"""
PITWALL — F1 Tire Degradation Predictor
fetch.py | Data Pipeline Module
Author: Manan Prajapati
"""

import fastf1
import pandas as pd
import os
import logging

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

CACHE_DIR = "data/cache"
RAW_DIR = "data/raw"

# Races to fetch (year, round_number, session_type)
SESSIONS_TO_FETCH = [
    (2023, "Bahrain Grand Prix", "R"),
    (2023, "Saudi Arabian Grand Prix", "R"),
    (2023, "Australian Grand Prix", "R"),
    (2023, "Monaco Grand Prix", "R"),
    (2023, "British Grand Prix", "R"),
    (2024, "Bahrain Grand Prix", "R"),
    (2024, "Saudi Arabian Grand Prix", "R"),
    (2024, "Australian Grand Prix", "R"),
]

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PITWALL.fetch")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

fastf1.Cache.enable_cache(CACHE_DIR)


# ─────────────────────────────────────────
# CORE FETCH FUNCTION
# ─────────────────────────────────────────

def fetch_session(year: int, grand_prix: str, session_type: str = "R") -> pd.DataFrame | None:
    """
    Fetch lap + tire data for a single F1 session using FastF1.

    Args:
        year        : Season year (e.g. 2023)
        grand_prix  : Race name (e.g. "Bahrain Grand Prix")
        session_type: 'R' = Race, 'Q' = Qualifying, 'FP1'/'FP2'/'FP3'

    Returns:
        DataFrame with laps + tire info, or None if fetch failed.
    """
    logger.info(f"Fetching → {year} | {grand_prix} | {session_type}")

    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load(laps=True, telemetry=False, weather=True, messages=False)

        laps = session.laps.copy()

        # ── Select relevant columns ──
        cols_needed = [
            "Driver", "Team", "LapNumber", "LapTime",
            "Stint", "Compound", "TyreLife",
            "FreshTyre", "TrackStatus", "IsAccurate",
        ]
        # Only keep columns that exist (FastF1 versions vary)
        cols_available = [c for c in cols_needed if c in laps.columns]
        laps = laps[cols_available].copy()

        # ── Basic cleaning ──
        laps = laps[laps["IsAccurate"] == True].copy() if "IsAccurate" in laps.columns else laps
        laps["LapTime_s"] = laps["LapTime"].dt.total_seconds() if "LapTime" in laps.columns else None
        laps.drop(columns=["LapTime", "IsAccurate"], errors="ignore", inplace=True)

        # ── Add metadata ──
        laps["Year"] = year
        laps["GrandPrix"] = grand_prix
        laps["SessionType"] = session_type

        logger.info(f"  ✓ {len(laps)} clean laps | Drivers: {laps['Driver'].nunique()}")
        return laps

    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        return None


# ─────────────────────────────────────────
# BATCH FETCH + SAVE
# ─────────────────────────────────────────

def fetch_all_sessions(sessions: list[tuple] = SESSIONS_TO_FETCH) -> pd.DataFrame:
    """
    Loops over all configured sessions, fetches data, concatenates into one DataFrame.
    Saves each session individually + a combined master CSV.
    """
    all_laps = []

    for year, gp, stype in sessions:
        df = fetch_session(year, gp, stype)
        if df is not None and not df.empty:
            # Save per-session file
            safe_name = gp.lower().replace(" ", "_")
            filename = f"{RAW_DIR}/{year}_{safe_name}_{stype}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"  Saved → {filename}")
            all_laps.append(df)

    if not all_laps:
        logger.warning("No data fetched. Check session names or FastF1 cache.")
        return pd.DataFrame()

    master_df = pd.concat(all_laps, ignore_index=True)
    master_path = f"{RAW_DIR}/master_laps.csv"
    master_df.to_csv(master_path, index=False)

    logger.info(f"\n{'─'*45}")
    logger.info(f"  PITWALL fetch complete.")
    logger.info(f"  Total laps     : {len(master_df)}")
    logger.info(f"  Unique races   : {master_df['GrandPrix'].nunique()}")
    logger.info(f"  Unique drivers : {master_df['Driver'].nunique()}")
    logger.info(f"  Saved to       : {master_path}")
    logger.info(f"{'─'*45}\n")

    return master_df


# ─────────────────────────────────────────
# QUICK PREVIEW UTILITY
# ─────────────────────────────────────────

def preview_data(df: pd.DataFrame, n: int = 5) -> None:
    """Print a quick preview of the fetched data."""
    if df.empty:
        print("No data to preview.")
        return

    print("\n── PITWALL Data Preview ──")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    print(df.head(n).to_string(index=False))
    print("\n── Compound Distribution ──")
    if "Compound" in df.columns:
        print(df["Compound"].value_counts().to_string())
    print("\n── Tyre Life Stats ──")
    if "TyreLife" in df.columns:
        print(df["TyreLife"].describe().round(2).to_string())


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("\n🏎  PITWALL | Data Fetch Pipeline\n")
    master = fetch_all_sessions()
    preview_data(master)