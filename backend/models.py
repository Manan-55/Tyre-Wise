"""
PITWALL — Database Models
models.py | SQLAlchemy Table Definitions
Tables: sessions, predictions, pit_events
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    ForeignKey, DateTime, Index
)
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


# ─────────────────────────────────────────
# TABLE 1: sessions
# One row per race session tracked
# ─────────────────────────────────────────

class Session(Base):
    __tablename__ = "sessions"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_key  = Column(Integer, unique=True, index=True)  # OpenF1 session ID
    meeting_name = Column(String)                            # "Miami Grand Prix"
    total_laps   = Column(Integer, default=0)
    created_at   = Column(DateTime, default=datetime.utcnow)

    # Relationships
    predictions  = relationship("Prediction", back_populates="session", cascade="all, delete")
    pit_events   = relationship("PitEvent", back_populates="session", cascade="all, delete")

    def __repr__(self):
        return f"<Session {self.meeting_name} ({self.session_key})>"


# ─────────────────────────────────────────
# TABLE 2: predictions
# One row per driver per lap — XGBoost output
# ─────────────────────────────────────────

class Prediction(Base):
    __tablename__ = "predictions"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    session_id           = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    driver_number        = Column(Integer, index=True)
    driver_name          = Column(String)                  # "Max VERSTAPPEN"
    abbreviation         = Column(String(3), index=True)   # "VER"
    team                 = Column(String)
    lap_number           = Column(Integer)
    compound             = Column(String(10))              # SOFT/MEDIUM/HARD
    tyre_life            = Column(Integer)
    stint_number         = Column(Integer)
    deg_rate             = Column(Float)
    stint_progress       = Column(Float)
    predicted_lap_time_s = Column(Float)                   # XGBoost output
    actual_lap_time_s    = Column(Float, nullable=True)    # from OpenF1 (filled after race)
    pit_recommended      = Column(Boolean, default=False)
    created_at           = Column(DateTime, default=datetime.utcnow)

    # Relationship
    session = relationship("Session", back_populates="predictions")

    # Composite index — fast queries by session + driver
    __table_args__ = (
        Index("idx_session_driver", "session_id", "abbreviation"),
        Index("idx_session_lap", "session_id", "lap_number"),
    )

    def __repr__(self):
        return f"<Prediction {self.abbreviation} Lap {self.lap_number} {self.compound}>"


# ─────────────────────────────────────────
# TABLE 3: pit_events
# Tracks pit recommendations vs actual pit stops
# ─────────────────────────────────────────

class PitEvent(Base):
    __tablename__ = "pit_events"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    session_id       = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    driver_number    = Column(Integer)
    abbreviation     = Column(String(3), index=True)
    lap_recommended  = Column(Integer)     # when PITWALL said pit
    lap_actual       = Column(Integer, nullable=True)  # when driver actually pitted
    compound_before  = Column(String(10))
    compound_after   = Column(String(10), nullable=True)
    accuracy_laps    = Column(Integer, nullable=True)  # difference (predicted vs actual)
    created_at       = Column(DateTime, default=datetime.utcnow)

    # Relationship
    session = relationship("Session", back_populates="pit_events")

    def __repr__(self):
        return f"<PitEvent {self.abbreviation} recommended={self.lap_recommended} actual={self.lap_actual}>"