"""
ATLAS-X Fraud Ring Checker
───────────────────────────
Lightweight Neo4j query layer used by both the FastAPI prediction
pipeline and the offline evaluation scripts.

Singleton pattern: call FraudRingChecker.get() to obtain the shared
instance. The driver is created lazily, so the module is safe to import
even when Neo4j is not running.

Usage:
    checker = FraudRingChecker.get()
    result  = checker.check(card_id)
"""
import os
import time
from functools import lru_cache
from typing import Optional

_INSTANCE: Optional["FraudRingChecker"] = None

NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password123")

# Minimum connected fraud-cards to declare a ring membership
RING_THRESHOLD = 2

# Cypher — uses fraud_txn > 0 (our actual node property) not is_fraud
_QUERY = """
MATCH (c:Card {card_id: $card_id})

OPTIONAL MATCH (c)-[:USED_DEVICE]->(d:Device)<-[:USED_DEVICE]-(o1:Card)
WHERE o1.fraud_txn > 0 AND o1.card_id <> $card_id
WITH c, count(DISTINCT o1) AS device_frauds

OPTIONAL MATCH (c)-[:HAS_EMAIL]->(e:Email)<-[:HAS_EMAIL]-(o2:Card)
WHERE o2.fraud_txn > 0 AND o2.card_id <> c.card_id
WITH c, device_frauds, count(DISTINCT o2) AS email_frauds

OPTIONAL MATCH (c)-[:BILLING_ADDR]->(a:Address)<-[:BILLING_ADDR]-(o3:Card)
WHERE o3.fraud_txn > 0 AND o3.card_id <> c.card_id

RETURN device_frauds,
       email_frauds,
       count(DISTINCT o3)                                            AS address_frauds,
       (device_frauds + email_frauds + count(DISTINCT o3))          AS total_connections
"""

# Batch variant — flags only on DEVICE connections (strong, specific signal).
# Email/address are too broad (shared domains inflate counts).
_BATCH_QUERY_DEVICE_ONLY = """
UNWIND $card_ids AS cid
MATCH (c:Card {card_id: cid})

OPTIONAL MATCH (c)-[:USED_DEVICE]->(d:Device)<-[:USED_DEVICE]-(o1:Card)
WHERE o1.fraud_txn > 0 AND o1.card_id <> cid
WITH cid, count(DISTINCT o1) AS device_frauds
WHERE device_frauds >= $threshold

RETURN cid           AS card_id,
       device_frauds,
       0             AS email_frauds,
       0             AS address_frauds,
       device_frauds AS total_connections
"""

# Batch variant using all three connection types (use only when the graph
# has fine-grained entities, not shared email domains / coarse addresses).
_BATCH_QUERY_ALL = """
UNWIND $card_ids AS cid
MATCH (c:Card {card_id: cid})

OPTIONAL MATCH (c)-[:USED_DEVICE]->(d:Device)<-[:USED_DEVICE]-(o1:Card)
WHERE o1.fraud_txn > 0 AND o1.card_id <> cid
WITH c, cid, count(DISTINCT o1) AS device_frauds

OPTIONAL MATCH (c)-[:HAS_EMAIL]->(e:Email)<-[:HAS_EMAIL]-(o2:Card)
WHERE o2.fraud_txn > 0 AND o2.card_id <> cid
WITH c, cid, device_frauds, count(DISTINCT o2) AS email_frauds

OPTIONAL MATCH (c)-[:BILLING_ADDR]->(a:Address)<-[:BILLING_ADDR]-(o3:Card)
WHERE o3.fraud_txn > 0 AND o3.card_id <> cid

WITH cid, device_frauds, email_frauds, count(DISTINCT o3) AS address_frauds
WHERE (device_frauds + email_frauds + address_frauds) >= $threshold

RETURN cid              AS card_id,
       device_frauds,
       email_frauds,
       address_frauds,
       (device_frauds + email_frauds + address_frauds) AS total_connections
"""


def _risk_score(total_connections: int) -> float:
    if total_connections >= 5:
        return 0.9
    if total_connections >= 3:
        return 0.7
    if total_connections >= 2:
        return 0.5
    if total_connections == 1:
        return 0.3
    return 0.0


def _ring_type(device: int, email: int, address: int) -> str:
    active = []
    if device  >= RING_THRESHOLD: active.append("device")
    if email   >= RING_THRESHOLD: active.append("email")
    if address >= RING_THRESHOLD: active.append("address")
    if len(active) > 1:
        return "multiple"
    return active[0] if active else "none"


class FraudRingChecker:
    """Thread-safe singleton that wraps Neo4j queries with an LRU cache."""

    def __init__(self) -> None:
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        # Warm-up: verify connectivity
        with self._driver.session() as s:
            s.run("RETURN 1").consume()

    # ── Singleton access ──────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> "FraudRingChecker":
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = cls()
        return _INSTANCE

    @classmethod
    def reset(cls) -> None:
        """Force re-initialisation (useful in tests)."""
        global _INSTANCE
        if _INSTANCE is not None:
            try:
                _INSTANCE._driver.close()
            except Exception:
                pass
            _INSTANCE = None

    # ── Single-card check (LRU cached) ────────────────────────────────────────

    def check(self, card_id: str) -> dict:
        return self._cached_check(card_id)

    @lru_cache(maxsize=16_384)
    def _cached_check(self, card_id: str) -> dict:
        t0 = time.perf_counter()
        with self._driver.session() as session:
            rec = session.run(_QUERY, card_id=card_id).single()

        latency_ms = (time.perf_counter() - t0) * 1000

        if rec is None:
            return {
                "in_fraud_ring": False,
                "ring_type": "none",
                "connected_frauds": 0,
                "graph_risk_score": 0.0,
                "latency_ms": round(latency_ms, 2),
            }

        device_f  = int(rec["device_frauds"]  or 0)
        email_f   = int(rec["email_frauds"]   or 0)
        address_f = int(rec["address_frauds"] or 0)
        total     = int(rec["total_connections"] or 0)

        in_ring  = total >= RING_THRESHOLD
        rtype    = _ring_type(device_f, email_f, address_f) if in_ring else "none"
        risk     = _risk_score(total)

        return {
            "in_fraud_ring":    in_ring,
            "ring_type":        rtype,
            "connected_frauds": total,
            "device_frauds":    device_f,
            "email_frauds":     email_f,
            "address_frauds":   address_f,
            "graph_risk_score": risk,
            "latency_ms":       round(latency_ms, 2),
        }

    # ── Batch check (no LRU — caller handles caching if needed) ──────────────

    def check_batch(
        self,
        card_ids: list[str],
        threshold: int = RING_THRESHOLD,
        device_only: bool = True,
        batch_size: int = 2000,
    ) -> dict[str, dict]:
        """
        Return {card_id: result} for cards that meet the ring threshold.
        Cards NOT in rings are omitted (treat as clean).

        device_only=True (default): flag only on shared-device connections.
            This avoids the email/address false-positive explosion caused by
            shared domains (gmail.com etc.) and coarse billing regions.
        device_only=False: flag on any connection type (use when your graph
            has truly unique email/address entities, not aggregated domains).
        """
        unique_ids = list(set(card_ids))
        flagged: dict[str, dict] = {}
        query = _BATCH_QUERY_DEVICE_ONLY if device_only else _BATCH_QUERY_ALL

        for i in range(0, len(unique_ids), batch_size):
            chunk = unique_ids[i : i + batch_size]
            with self._driver.session() as session:
                rows = session.run(query, card_ids=chunk, threshold=threshold)
                for r in rows:
                    d   = int(r["device_frauds"]  or 0)
                    e   = int(r["email_frauds"]   or 0)
                    a   = int(r["address_frauds"] or 0)
                    tot = int(r["total_connections"] or 0)
                    flagged[r["card_id"]] = {
                        "in_fraud_ring":    True,
                        "ring_type":        _ring_type(d, e, a),
                        "connected_frauds": tot,
                        "device_frauds":    d,
                        "email_frauds":     e,
                        "address_frauds":   a,
                        "graph_risk_score": _risk_score(tot),
                    }

        return flagged

    def close(self) -> None:
        self._driver.close()
