import hashlib
import hmac
import json
import redis.asyncio as aioredis

from fastapi import APIRouter, Request, HTTPException, status
from config import settings
from ingest.normalizer import normalize
from enrichment.geoip import enrich_geoip
from enrichment.mitre import enrich_mitre
from enrichment.scorer import compute_threat_score
from database.elastic import index_event, upsert_ioc
from database.neo4j import write_session_graph
from database.postgres import AsyncSessionLocal, Session, Alert, AlertStatus
from sqlalchemy import select

router = APIRouter()
_redis = None


def get_redis():
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def verify_hmac(payload: bytes, signature: str) -> bool:
    expected = hmac.new(
        settings.hmac_secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


async def process_event(raw: dict):
    """Full pipeline: normalize → enrich → score → store → publish."""

    # 1. Normalize
    normalized = normalize(raw)
    event = normalized.model_dump()

    # 2. Enrich
    event = enrich_geoip(event)
    event = enrich_mitre(event)
    event = compute_threat_score(event)

    # Serialise the timestamp only once scoring has run. This used to happen
    # immediately after model_dump(), which left compute_threat_score comparing
    # a str where it expected a datetime — so its off-hours branch was
    # unreachable and the boost silently never applied. Elasticsearch and Redis
    # are the only consumers that need a string, and both come after here.
    event["timestamp"] = normalized.timestamp.isoformat()

    # 3. Store in Elasticsearch
    await index_event(event)

    # 4. Upsert IOCs
    await upsert_ioc("ip", event["src_ip"], event)
    if event.get("hassh"):
        await upsert_ioc("hassh", event["hassh"], event)
    if event.get("username"):
        await upsert_ioc("username", event["username"], event)

    # 5. Write to Neo4j graph
    await write_session_graph(event)

    # 6. Upsert session in Postgres
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Session).where(Session.session_id == event["session_id"])
        )
        sess = result.scalar_one_or_none()
        if not sess:
            sess = Session(
                session_id  = event["session_id"],
                src_ip      = event["src_ip"],
                sensor_id   = event["sensor_id"],
                protocol    = event.get("protocol", "ssh"),
                started_at  = normalized.timestamp.replace(tzinfo=None),
                hassh       = event.get("hassh"),
                ssh_version = event.get("ssh_version"),
                country     = event.get("country"),
                city        = event.get("city"),
                threat_score = event.get("threat_score", 0),
            )
            db.add(sess)
        else:
            sess.event_count  += 1
            sess.threat_score  = max(sess.threat_score or 0, event.get("threat_score", 0))
            if event.get("event_type") == "brute_force":
                sess.login_attempts = (sess.login_attempts or 0) + 1
            if event.get("event_type") == "session_end" and event.get("duration"):
                sess.ended_at = normalized.timestamp.replace(tzinfo=None)
                sess.duration = event.get("duration")
            if event.get("hassh") and not sess.hassh:
                sess.hassh = event["hassh"]
            if event.get("country") and not sess.country:
                sess.country = event["country"]

        # 7. Auto-alert on high threat score
        if event.get("threat_score", 0) >= 70:
            alert = Alert(
                session_id      = event["session_id"],
                src_ip          = event["src_ip"],
                alert_type      = event.get("event_type", "unknown"),
                description     = event.get("raw_message", ""),
                threat_score    = event.get("threat_score", 0),
                mitre_technique = event.get("mitre_technique"),
                country         = event.get("country"),
                status          = AlertStatus.open,
            )
            db.add(alert)

        await db.commit()

    # 8. Publish to Redis for WebSocket live feed
    r = get_redis()
    await r.publish("events:live", json.dumps(event, default=str))

    return event


@router.post("/")
async def ingest_log(request: Request):
    """
    Accepts a single Cowrie log line (JSON).
    Zero Trust layer sends HMAC signature in X-Signature header.
    For testing without ZT layer, signature check is skippable via env.
    """
    body = await request.body()

    # HMAC verification — skip if secret is set to 'dev'
    if settings.hmac_secret != "dev":
        signature = request.headers.get("X-Signature", "")
        if not verify_hmac(body, signature):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid HMAC signature"
            )

    try:
        raw = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON"
        )

    event = await process_event(raw)
    return {"status": "ok", "threat_score": event.get("threat_score"), "session": event.get("session_id")}


@router.post("/batch")
async def ingest_batch(request: Request):
    """Operator upload path: multiple Cowrie log lines as NDJSON.

    This exists for the dashboard's Ingest Logs page, which lets an analyst
    upload a Cowrie JSONL file by hand. A browser cannot hold the signing
    secret, so this endpoint cannot require a signature the way /ingest/ does.

    Honeypot sensors do NOT use this path. Sidecars stream single signed
    events to /ingest/, where the signature is always verified — so the
    zero-trust guarantee covers every event that arrives from a node.

    Guarded by `allow_unsigned_batch`, which must be set false before the hub
    is exposed anywhere; otherwise anyone who can reach it can inject events.
    """
    if not settings.allow_unsigned_batch:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Unsigned batch ingestion is disabled. Sensors should stream "
                "signed events to POST /ingest/ instead."
            ),
        )

    body = await request.body()
    lines = [l.strip() for l in body.decode().splitlines() if l.strip()]
    results = []
    for line in lines:
        try:
            raw = json.loads(line)
            event = await process_event(raw)
            results.append({"session": event.get("session_id"), "score": event.get("threat_score")})
        except Exception as e:
            results.append({"error": str(e)})
    return {"processed": len(results), "results": results}