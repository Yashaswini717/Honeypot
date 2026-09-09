from datetime import datetime, timezone


def _as_datetime(value) -> datetime | None:
    """Read a timestamp that may already have been serialised for storage.

    The ingest pipeline hands us a datetime, but it stringifies the field
    before Elasticsearch and Redis see it. When those two steps were ordered
    the other way round this function's caller silently lost its off-hours
    branch for every event ever ingested — nothing raised, the score was just
    quietly 5 lower. Accepting both shapes means the ordering can change again
    without taking the boost down with it.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def compute_threat_score(event: dict) -> dict:
    score = event.get("base_score", 10)

    # boost for high-value techniques
    technique = event.get("mitre_technique")
    if technique in ("T1078", "T1059", "T1041"):
        score = min(score + 20, 100)

    # boost if login succeeded
    if event.get("event_type") == "compromise":
        score = 95

    # boost for command execution
    if event.get("command"):
        score = min(score + 15, 100)

    # slight boost for off-hours (UTC 0–6)
    try:
        ts = _as_datetime(event.get("timestamp"))
        if ts is not None:
            # A naive timestamp is already UTC here — the sensors emit
            # ISO-8601 UTC with a trailing Z — but astimezone() would read it
            # as local time and shift the hour on any non-UTC host.
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            hour = ts.astimezone(timezone.utc).hour
            if 0 <= hour < 6:
                score = min(score + 5, 100)
    except Exception:
        pass

    event["threat_score"] = score
    return event