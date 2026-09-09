"""Candidate decoy actions the adaptive engine can choose between per intent.

The `INTENT_TO_ACTION` map this replaced picked exactly one action per intent —
there was nothing to *choose* between, so nothing could be learned. Here each
intent gets several candidate strategies; the
`AdaptiveDecisionEngine` (see `core/adaptive_engine.py`) learns, per intent,
which candidate tends to keep an attacker engaged (or trips a honeytoken)
most often, via Thompson Sampling.

Each action also maps to an existing `populator.strategies.PopulationStrategy`
profile so a caller (honeypot node, orchestration script) can turn the
chosen action into real generated content with the endpoints that already
exist: `POST /api/v1/populate/{honeypot_id}/profile/{profile_name}`.
"""

from __future__ import annotations

from typing import Final

from core.intent_taxonomy import INTENT_CLASSES


# Candidate decoy strategies per intent. The first entry in each list is the
# team's original Phase-3 default, carried over from the static map this
# replaced; the rest are alternatives the bandit is free to discover are better.
ACTION_CATALOG: Final[dict[str, list[str]]] = {
    "reconnaissance": [
        "show_fake_endpoints",
        "populate_developer_workstation",
        "serve_minimal_banner",
    ],
    "privilege_escalation": [
        "inject_fake_sudo_config",
        "plant_honeytoken_credentials",
        "simulate_privileged_process_list",
    ],
    "persistence": [
        "simulate_cron_jobs",
        "expose_fake_authorized_keys",
        "populate_production_server",
    ],
    "lateral_movement": [
        "expose_fake_internal_ips",
        "serve_fake_network_map",
        "plant_ssh_honeytoken",
    ],
    "data_exfiltration": [
        "serve_fake_sensitive_files",
        "plant_tracked_honeytoken_archive",
        "populate_database_server",
    ],
}

# Fallback candidates for any intent that isn't in the catalog (defensive;
# INTENT_CLASSES should always cover this in practice).
DEFAULT_ACTIONS: Final[list[str]] = ["show_fake_endpoints", "populate_developer_workstation"]

assert set(ACTION_CATALOG) == set(INTENT_CLASSES), "ACTION_CATALOG must cover every intent class"


# Maps each action label to an existing populate profile, so a chosen action
# can be executed immediately via the current populator without inventing a
# new content-generation path. A few actions share a base profile but are
# given genuinely distinct generated content within it (see the
# `context.get("action")` branches in populator/strategies.py) — every
# action within a given intent's own candidate list produces different
# deployed content, since that's the only comparison the bandit actually
# makes.
ACTION_TO_PROFILE: Final[dict[str, str]] = {
    "show_fake_endpoints": "web_server",
    "populate_developer_workstation": "developer_workstation",
    "serve_minimal_banner": "developer_workstation",
    "inject_fake_sudo_config": "developer_workstation",
    "plant_honeytoken_credentials": "production_server",
    "simulate_privileged_process_list": "web_server",
    "simulate_cron_jobs": "production_server",
    "expose_fake_authorized_keys": "developer_workstation",
    "populate_production_server": "production_server",
    "expose_fake_internal_ips": "web_server",
    "serve_fake_network_map": "production_server",
    "plant_ssh_honeytoken": "developer_workstation",
    "serve_fake_sensitive_files": "database_server",
    "plant_tracked_honeytoken_archive": "production_server",
    "populate_database_server": "database_server",
}


def get_candidate_actions(intent: str) -> list[str]:
    """Return the candidate decoy actions the bandit may pick from for `intent`."""
    return ACTION_CATALOG.get(intent, DEFAULT_ACTIONS)
