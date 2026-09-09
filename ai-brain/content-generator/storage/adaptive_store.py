"""Persistence layer for the adaptive decision engine (contextual bandit).

Mirrors the style of `storage/honeytoken_store.py`: a small SQLAlchemy-backed
store, one class, explicit sessions per call, pydantic response models for
anything handed back outside the `with` block.

Two tables:
- `adaptive_arms`      one row per (intent, action) with a Beta(alpha, beta)
                        posterior over "does this action work for this intent".
- `adaptive_decisions` one row per action actually selected for a session,
                        pending until a reward is observed and it is resolved.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from config.logging_config import LoggerMixin
from config.settings import settings
from core.exceptions import DatabaseError
from core.utils import generate_unique_id

from .models import (
    AdaptiveArmDB,
    AdaptiveArmStats,
    AdaptiveDecisionDB,
    AdaptiveDecisionResponse,
    Base,
    utcnow_naive,
)


class AdaptiveLearningStore(LoggerMixin):
    """Store and update the contextual bandit's state."""

    @staticmethod
    def _create_engine(database_url: str):
        """Create a database engine with sane defaults for SQLite files."""
        engine_kwargs = {"echo": settings.database_echo}

        if database_url.startswith("sqlite:///"):
            db_path = database_url.removeprefix("sqlite:///")
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                engine_kwargs["poolclass"] = NullPool

        return create_engine(database_url, **engine_kwargs)

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize the adaptive learning store.

        Args:
            database_url: Database connection URL (uses settings if not provided)
        """
        self.database_url = database_url or settings.database_url
        self.engine = self._create_engine(self.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

        Base.metadata.create_all(self.engine)

        self.logger.info("adaptive_learning_store_initialized", database_url=self.database_url)

    @staticmethod
    def _to_arm_stats(arm: AdaptiveArmDB) -> AdaptiveArmStats:
        total = arm.alpha + arm.beta
        posterior_mean = arm.alpha / total if total > 0 else 0.0
        return AdaptiveArmStats(
            intent=arm.intent,
            action=arm.action,
            alpha=arm.alpha,
            beta=arm.beta,
            posterior_mean=posterior_mean,
            times_selected=arm.times_selected,
            total_reward=arm.total_reward,
            updated_at=arm.updated_at,
        )

    def _get_or_create_arm(self, session, intent: str, action: str) -> AdaptiveArmDB:
        """Fetch an arm row, creating it with a uniform Beta(1, 1) prior if new."""
        stmt = select(AdaptiveArmDB).where(
            AdaptiveArmDB.intent == intent,
            AdaptiveArmDB.action == action,
        )
        arm = session.execute(stmt).scalar_one_or_none()
        if arm is None:
            arm = AdaptiveArmDB(intent=intent, action=action, alpha=1.0, beta=1.0)
            session.add(arm)
            session.flush()
        return arm

    def select_action(
        self,
        intent: str,
        candidate_actions: list[str],
        session_id: Optional[str] = None,
        honeypot_id: Optional[str] = None,
    ) -> tuple[str, AdaptiveArmStats, Optional[str]]:
        """
        Thompson-sample an action for `intent` among `candidate_actions`.

        For every candidate, draw theta ~ Beta(alpha, beta) from its current
        posterior and pick the candidate with the highest draw. This
        naturally balances exploration (arms with little evidence have wide,
        uncertain posteriors and occasionally win by chance) against
        exploitation (arms with a strong track record usually win).

        If `session_id` is given, the selection is persisted as a pending
        decision so it can be resolved later via `resolve_decision`. Without
        a session_id (stateless/ad-hoc calls) the pick still reflects the
        current posterior, but nothing is recorded to resolve.

        Returns:
            (chosen_action, arm_stats_after_selection, decision_id_or_None)
        """
        if not candidate_actions:
            raise ValueError(f"No candidate actions provided for intent={intent!r}")

        try:
            with self.SessionLocal() as session:
                best_action: Optional[str] = None
                best_theta = -1.0
                best_arm: Optional[AdaptiveArmDB] = None

                for action in candidate_actions:
                    arm = self._get_or_create_arm(session, intent, action)
                    theta = random.betavariate(arm.alpha, arm.beta)
                    if theta > best_theta:
                        best_theta = theta
                        best_action = action
                        best_arm = arm

                assert best_action is not None and best_arm is not None  # candidate_actions non-empty

                best_arm.times_selected += 1
                best_arm.updated_at = utcnow_naive()

                decision_id: Optional[str] = None
                if session_id:
                    decision_id = generate_unique_id()
                    session.add(
                        AdaptiveDecisionDB(
                            decision_id=decision_id,
                            session_id=session_id,
                            honeypot_id=honeypot_id,
                            intent=intent,
                            action=best_action,
                        )
                    )

                session.commit()
                session.refresh(best_arm)

                stats = self._to_arm_stats(best_arm)

                self.logger.info(
                    "adaptive_action_selected",
                    intent=intent,
                    action=best_action,
                    theta=round(best_theta, 4),
                    posterior_mean=round(stats.posterior_mean, 4),
                    times_selected=stats.times_selected,
                    session_id=session_id,
                    decision_id=decision_id,
                )

                return best_action, stats, decision_id
        except Exception as e:
            self.logger.error("adaptive_action_selection_failed", intent=intent, error=str(e))
            raise DatabaseError(f"Failed to select adaptive action: {e}") from e

    def get_pending_decision(self, session_id: str) -> Optional[AdaptiveDecisionResponse]:
        """Return the most recent unresolved decision for a session, if any."""
        try:
            with self.SessionLocal() as session:
                stmt = (
                    select(AdaptiveDecisionDB)
                    .where(
                        AdaptiveDecisionDB.session_id == session_id,
                        AdaptiveDecisionDB.resolved == False,  # noqa: E712
                    )
                    .order_by(AdaptiveDecisionDB.selected_at.desc())
                    .limit(1)
                )
                row = session.execute(stmt).scalar_one_or_none()
                if row is None:
                    return None
                return AdaptiveDecisionResponse.model_validate(row)
        except Exception as e:
            self.logger.error("adaptive_pending_lookup_failed", session_id=session_id, error=str(e))
            raise DatabaseError(f"Failed to look up pending decision: {e}") from e

    def resolve_decision(
        self,
        decision_id: str,
        reward: float,
        reason: str = "auto",
    ) -> Optional[AdaptiveArmStats]:
        """
        Resolve a pending decision with an observed reward in [0, 1] and
        update the corresponding arm's Beta posterior:

            alpha += reward
            beta  += (1 - reward)

        A reward of 1.0 (e.g. a honeytoken was triggered) pushes the arm's
        posterior mean up; a reward of 0.0 (e.g. the session went cold
        immediately) pushes it down. Returns None if the decision does not
        exist or was already resolved.
        """
        clamped_reward = max(0.0, min(1.0, reward))

        try:
            with self.SessionLocal() as session:
                stmt = select(AdaptiveDecisionDB).where(
                    AdaptiveDecisionDB.decision_id == decision_id,
                    AdaptiveDecisionDB.resolved == False,  # noqa: E712
                )
                decision = session.execute(stmt).scalar_one_or_none()
                if decision is None:
                    return None

                decision.resolved = True
                decision.resolved_at = utcnow_naive()
                decision.reward = clamped_reward
                decision.resolution_reason = reason

                arm = self._get_or_create_arm(session, decision.intent, decision.action)
                arm.alpha += clamped_reward
                arm.beta += 1.0 - clamped_reward
                arm.total_reward += clamped_reward
                arm.updated_at = utcnow_naive()

                session.commit()
                session.refresh(arm)

                stats = self._to_arm_stats(arm)

                self.logger.info(
                    "adaptive_decision_resolved",
                    decision_id=decision_id,
                    intent=decision.intent,
                    action=decision.action,
                    reward=clamped_reward,
                    reason=reason,
                    posterior_mean=round(stats.posterior_mean, 4),
                )

                return stats
        except Exception as e:
            self.logger.error("adaptive_decision_resolution_failed", decision_id=decision_id, error=str(e))
            raise DatabaseError(f"Failed to resolve adaptive decision: {e}") from e

    def resolve_pending_for_session(
        self,
        session_id: str,
        reward: float,
        reason: str = "explicit_feedback",
    ) -> list[AdaptiveArmStats]:
        """Resolve every still-pending decision for a session (used by /adaptive/feedback)."""
        resolved: list[AdaptiveArmStats] = []
        pending = self.get_pending_decision(session_id)
        while pending is not None:
            stats = self.resolve_decision(pending.decision_id, reward, reason=reason)
            if stats is None:
                break
            resolved.append(stats)
            pending = self.get_pending_decision(session_id)
        return resolved

    def get_arm_stats(self, intent: Optional[str] = None) -> list[AdaptiveArmStats]:
        """Return current posterior stats for all arms (optionally filtered by intent)."""
        try:
            with self.SessionLocal() as session:
                stmt = select(AdaptiveArmDB)
                if intent:
                    stmt = stmt.where(AdaptiveArmDB.intent == intent)
                stmt = stmt.order_by(AdaptiveArmDB.intent, AdaptiveArmDB.action)
                rows = session.execute(stmt).scalars().all()
                return [self._to_arm_stats(r) for r in rows]
        except Exception as e:
            self.logger.error("adaptive_arm_stats_failed", error=str(e))
            raise DatabaseError(f"Failed to get adaptive arm stats: {e}") from e

    def reset_all(self) -> int:
        """Delete all bandit state (dev/demo utility). Returns number of arms removed."""
        try:
            with self.SessionLocal() as session:
                deleted_arms = session.query(AdaptiveArmDB).delete()
                session.query(AdaptiveDecisionDB).delete()
                session.commit()
                self.logger.warning("adaptive_state_reset", arms_deleted=deleted_arms)
                return deleted_arms
        except Exception as e:
            self.logger.error("adaptive_state_reset_failed", error=str(e))
            raise DatabaseError(f"Failed to reset adaptive state: {e}") from e

    def close(self) -> None:
        """Dispose engine resources."""
        self.engine.dispose()
