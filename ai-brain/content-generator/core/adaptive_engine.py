"""The learning decision engine.

This replaced a static `DecisionEngine` whose `suggest_action()` was a plain
lookup, `INTENT_TO_ACTION[intent]`: same intent, same action, forever — nothing
about it changed based on how attackers actually responded, so despite the name
it never adapted. That module has been deleted rather than left beside this one;
two engines with near-identical names, only one of them wired up, is a trap.

`AdaptiveDecisionEngine` keeps a Thompson-Sampling contextual bandit per
intent (state) over several candidate decoy actions (see
`core.adaptive_actions.ACTION_CATALOG`), and updates its belief about which
action works best for each intent from real feedback — honeytoken hits and
kill-chain progression (see `core.reward`). It is the "Decision & Strategy
Engine" box in the project's architecture diagram, wired to the "Adaptive
Decoys" feedback loop rather than a fixed switch statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.adaptive_actions import get_candidate_actions
from core.intent_taxonomy import INTENT_CLASSES
from core.reward import RewardCalculator
from storage.adaptive_store import AdaptiveLearningStore
from storage.models import AdaptiveArmStats


@dataclass
class AdaptiveActionSuggestion:
    """What the bandit chose, plus the posterior state that explains the choice."""

    intent: str
    action: str
    decision_id: Optional[str]
    posterior_mean: float
    times_selected: int


class AdaptiveDecisionEngine:
    """Learning decision engine: picks a decoy action per intent and improves from feedback."""

    def __init__(
        self,
        store: Optional[AdaptiveLearningStore] = None,
        reward_calculator: Optional[RewardCalculator] = None,
    ):
        self.store = store or AdaptiveLearningStore()
        self.reward_calculator = reward_calculator or RewardCalculator()

    def decide(
        self,
        intent: str,
        session_id: Optional[str] = None,
        honeypot_id: Optional[str] = None,
    ) -> AdaptiveActionSuggestion:
        """
        Choose a decoy action for `intent`.

        If `session_id` was seen before with a still-pending decision (i.e.
        this is a later turn in the same attacker session), that earlier
        decision is resolved first — using this new intent as the "what
        happened next" evidence — before a new action is chosen.
        """
        if intent not in INTENT_CLASSES:
            raise ValueError(f"Unsupported intent: {intent}")

        if session_id:
            self._resolve_previous_decision(session_id, honeypot_id, new_intent=intent)

        candidates = get_candidate_actions(intent)
        action, stats, decision_id = self.store.select_action(
            intent=intent,
            candidate_actions=candidates,
            session_id=session_id,
            honeypot_id=honeypot_id,
        )

        return AdaptiveActionSuggestion(
            intent=intent,
            action=action,
            decision_id=decision_id,
            posterior_mean=stats.posterior_mean,
            times_selected=stats.times_selected,
        )

    def _resolve_previous_decision(
        self,
        session_id: str,
        honeypot_id: Optional[str],
        new_intent: str,
    ) -> None:
        pending = self.store.get_pending_decision(session_id)
        if pending is None:
            return

        breakdown = self.reward_calculator.compute(
            previous_intent=pending.intent,
            new_intent=new_intent,
            honeypot_id=pending.honeypot_id or honeypot_id,
            since=pending.selected_at,
        )
        self.store.resolve_decision(pending.decision_id, breakdown.reward, reason=breakdown.reason)

    def resolve_feedback(
        self,
        session_id: str,
        reward: float,
        reason: str = "explicit_feedback",
    ) -> int:
        """
        Resolve all still-pending decisions for a session with an explicit
        reward (used for signals `decide()` can't infer on its own, e.g. a
        session disconnecting). Returns the number of decisions resolved.
        """
        resolved = self.store.resolve_pending_for_session(session_id, reward, reason=reason)
        return len(resolved)

    def get_stats(self, intent: Optional[str] = None) -> list[AdaptiveArmStats]:
        """Current posterior stats for every arm (optionally filtered to one intent)."""
        return self.store.get_arm_stats(intent=intent)
