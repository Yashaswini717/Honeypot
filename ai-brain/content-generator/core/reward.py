"""Turns attacker behaviour into a reward signal for the adaptive bandit.

The bandit (see `core/adaptive_engine.py`, `storage/adaptive_store.py`) needs
a number in [0, 1] to say "how well did that decoy action work". This module
computes it from two things the system already tracks, so nothing new has to
be instrumented on the honeypot side:

1. Honeytoken access — `HoneytokenDB.access_count` / `accessed_at` already
   record when an attacker touches a planted credential/file. That's about
   as strong a positive signal as a honeypot ever gets.
2. Kill-chain progression — did the attacker's classified intent move
   deeper along the kill chain after this decoy was shown (recon -> privesc
   -> ... -> exfil), stay at the same stage (still engaged), or drop off?
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from core.intent_taxonomy import KILL_CHAIN_ORDER
from storage.honeytoken_store import HoneytokenStore

REWARD_HONEYTOKEN_TRIGGERED = 1.0
REWARD_ESCALATED = 0.7
REWARD_SESSION_CONTINUED = 0.4
REWARD_SESSION_DEESCALATED = 0.2
REWARD_SESSION_TERMINATED = 0.0


@dataclass
class RewardBreakdown:
    reward: float
    reason: str


def progression_reward(previous_intent: str, new_intent: str) -> RewardBreakdown:
    """Reward derived purely from where the attacker's intent moved next."""
    prev_depth = KILL_CHAIN_ORDER.get(previous_intent, 0)
    new_depth = KILL_CHAIN_ORDER.get(new_intent, 0)

    if new_depth > prev_depth:
        return RewardBreakdown(REWARD_ESCALATED, "intent_escalated")
    if new_depth == prev_depth:
        return RewardBreakdown(REWARD_SESSION_CONTINUED, "session_continued")
    return RewardBreakdown(REWARD_SESSION_DEESCALATED, "intent_deescalated")


class RewardCalculator:
    """
    Computes a [0, 1] reward for a resolved decision.

    `honeytoken_store` is optional: without it (or without a `honeypot_id`)
    the calculator still works, it just falls back to kill-chain-progression
    reward only.
    """

    def __init__(self, honeytoken_store: Optional[HoneytokenStore] = None):
        self.honeytoken_store = honeytoken_store

    def compute(
        self,
        previous_intent: str,
        new_intent: str,
        honeypot_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> RewardBreakdown:
        """
        Reward for the decoy action that was in effect between `since` and
        now, given what the attacker did next.

        A triggered honeytoken always wins (it's unambiguous ground truth);
        otherwise the reward falls back to kill-chain progression.
        """
        if self._honeytoken_triggered(honeypot_id, since):
            return RewardBreakdown(REWARD_HONEYTOKEN_TRIGGERED, "honeytoken_triggered")

        return progression_reward(previous_intent, new_intent)

    def _honeytoken_triggered(self, honeypot_id: Optional[str], since: Optional[datetime]) -> bool:
        if not self.honeytoken_store or not honeypot_id:
            return False
        try:
            tokens = self.honeytoken_store.list_honeytokens(
                honeypot_id=honeypot_id, active_only=False, limit=200
            )
        except Exception:
            # Reward calculation must never break the decision path.
            return False

        for token in tokens:
            if token.accessed_at is None:
                continue
            # Both sides of this comparison are naive UTC: `since` is a
            # decision's selected_at, defaulted from the database's func.now(),
            # and accessed_at is written through storage.models.utcnow_naive().
            # They used to disagree — accessed_at was local time — which skewed
            # this test by the host's UTC offset in whichever direction that
            # offset ran: ahead of UTC it returned True for accesses that
            # predated the decision, paying full reward for decoys nobody
            # touched. Keep any new writer on utcnow_naive() too.
            if since is None or token.accessed_at >= since:
                return True
        return False
