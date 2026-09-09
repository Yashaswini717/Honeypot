from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, JSON, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


def utcnow_naive() -> datetime:
    """The one way this schema writes a timestamp from Python.

    Every DateTime column here is naive, and every column-level default is
    `func.now()` — the *database's* clock, which is UTC. Anything Python writes
    into the same columns has to agree with that, or timestamps from the two
    sources cannot be compared. `datetime.now()` does not agree: it is local,
    so on any non-UTC host it silently shifts rows by the offset.

    That is not hypothetical here. `RewardCalculator` decides whether a
    honeytoken was touched after a bandit decision by comparing a Python-written
    `accessed_at` against a database-written `selected_at`, and the skew goes
    whichever way the host's offset does: east of Greenwich the Python-written
    side reads ahead, so stale accesses look like fresh ones and the bandit is
    handed rewards it did not earn; west of it, real hits are missed. This
    machine is UTC+5:30, so it had the first failure mode.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


# SQLAlchemy Models
class HoneytokenDB(Base):
    """Database model for honeytokens."""

    __tablename__ = "honeytokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    token_id = Column(String(100), unique=True, nullable=False, index=True)
    token_type = Column(String(50), nullable=False)
    token_value = Column(Text, nullable=False)
    honeypot_id = Column(String(100), index=True)
    file_path = Column(String(500))
    created_at = Column(DateTime, default=func.now(), nullable=False)
    accessed_at = Column(DateTime)
    access_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    token_metadata = Column(JSON)


class GenerationLogDB(Base):
    """Database model for generation logs."""

    __tablename__ = "generation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    generation_id = Column(String(100), unique=True, nullable=False, index=True)
    content_type = Column(String(50), nullable=False)
    file_type = Column(String(50))
    honeypot_id = Column(String(100), index=True)
    prompt_hash = Column(String(64))
    validation_score = Column(Float)
    is_valid = Column(Boolean)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    generation_time_ms = Column(Integer)
    token_metadata = Column(JSON)


# Pydantic Models
class HoneytokenCreate(BaseModel):
    """Pydantic model for creating honeytoken."""

    token_type: str
    token_value: str
    honeypot_id: Optional[str] = None
    file_path: Optional[str] = None
    token_metadata: dict = Field(default_factory=dict)


class HoneytokenResponse(BaseModel):
    """Pydantic model for honeytoken response."""

    id: int
    token_id: str
    token_type: str
    token_value: str
    honeypot_id: Optional[str]
    file_path: Optional[str]
    created_at: datetime
    accessed_at: Optional[datetime]
    access_count: int
    is_active: bool
    token_metadata: dict

    class Config:
        from_attributes = True


class HoneytokenAccessLog(BaseModel):
    """Log when honeytoken is accessed."""

    token_id: str
    accessed_at: datetime = Field(default_factory=utcnow_naive)
    access_source: Optional[str] = None
    access_metadata: dict = Field(default_factory=dict)


class GenerationLogCreate(BaseModel):
    """Pydantic model for creating generation log."""

    content_type: str
    file_type: str
    honeypot_id: Optional[str] = None
    prompt_hash: str
    validation_score: float
    is_valid: bool
    generation_time_ms: int
    token_metadata: dict = Field(default_factory=dict)


class GenerationLogResponse(BaseModel):
    """Pydantic model for generation log response."""

    id: int
    generation_id: str
    content_type: str
    file_type: str
    honeypot_id: Optional[str]
    prompt_hash: str
    validation_score: float
    is_valid: bool
    created_at: datetime
    generation_time_ms: int
    token_metadata: dict

    class Config:
        from_attributes = True


# ──────────────────────────────────────────────
# Adaptive learning (contextual bandit) models
# ──────────────────────────────────────────────


class AdaptiveArmDB(Base):
    """
    One row per (intent, action) 'arm' of the contextual bandit.

    alpha/beta are the Beta-distribution posterior parameters. They start at
    (1, 1) — a uniform prior, i.e. "no evidence yet" — and are nudged by
    every resolved decision: alpha += reward, beta += (1 - reward), where
    reward is clamped to [0, 1]. The posterior mean alpha / (alpha + beta)
    is the arm's current estimated success rate.
    """

    __tablename__ = "adaptive_arms"

    id = Column(Integer, primary_key=True, autoincrement=True)
    intent = Column(String(50), nullable=False, index=True)
    action = Column(String(100), nullable=False)
    alpha = Column(Float, default=1.0, nullable=False)
    beta = Column(Float, default=1.0, nullable=False)
    times_selected = Column(Integer, default=0, nullable=False)
    total_reward = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("intent", "action", name="uq_adaptive_arm_intent_action"),
    )


class AdaptiveDecisionDB(Base):
    """
    One row per action the bandit selected for a session. Created 'pending'
    (resolved=False) when the action is chosen, then resolved later once a
    reward signal is available (next intent-classify call for the same
    session, or an explicit /adaptive/feedback call).
    """

    __tablename__ = "adaptive_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    decision_id = Column(String(100), unique=True, nullable=False, index=True)
    session_id = Column(String(200), nullable=False, index=True)
    honeypot_id = Column(String(100))
    intent = Column(String(50), nullable=False)
    action = Column(String(100), nullable=False)
    selected_at = Column(DateTime, default=func.now(), nullable=False)
    resolved = Column(Boolean, default=False, nullable=False, index=True)
    resolved_at = Column(DateTime)
    reward = Column(Float)
    resolution_reason = Column(String(100))


class AdaptiveArmStats(BaseModel):
    """Pydantic view of a bandit arm, for the /adaptive/stats endpoint."""

    intent: str
    action: str
    alpha: float
    beta: float
    posterior_mean: float
    times_selected: int
    total_reward: float
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AdaptiveDecisionResponse(BaseModel):
    """Pydantic view of a persisted bandit decision."""

    decision_id: str
    session_id: str
    honeypot_id: Optional[str]
    intent: str
    action: str
    selected_at: datetime
    resolved: bool
    resolved_at: Optional[datetime]
    reward: Optional[float]
    resolution_reason: Optional[str]

    class Config:
        from_attributes = True
