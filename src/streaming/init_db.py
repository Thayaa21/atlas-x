"""
ATLAS-X – Postgres Schema Initialiser
───────────────────────────────────────
Creates the tables used by the streaming pipeline.

Tables:
  predictions   – every transaction scored by the consumer
  fraud_alerts  – FLAG and BLOCK events (subset of predictions)

Usage:
    python -m src.streaming.init_db

Environment:
    POSTGRES_DSN  default: postgresql://postgres:postgres@localhost:5432/atlasx
"""
import asyncio
import os

import asyncpg

POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://postgres:postgres@localhost:5432/atlasx",
)

DDL = """
-- ── predictions: every scored transaction ──────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id                BIGSERIAL PRIMARY KEY,
    transaction_id    VARCHAR(255) UNIQUE,
    fraud_probability DOUBLE PRECISION,
    graph_risk_score  DOUBLE PRECISION,
    decision          VARCHAR(20),
    customer_segment  VARCHAR(20),
    latency_ms        DOUBLE PRECISION,
    created_at        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_decision
    ON predictions (decision);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at
    ON predictions (created_at DESC);

-- ── fraud_alerts: FLAG + BLOCK events ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fraud_alerts (
    id                BIGSERIAL PRIMARY KEY,
    transaction_id    VARCHAR(255) UNIQUE,
    fraud_probability DOUBLE PRECISION,
    graph_risk_score  DOUBLE PRECISION,
    decision          VARCHAR(20),
    customer_segment  VARCHAR(20),
    is_fraud_label    INTEGER,          -- ground-truth label if available
    created_at        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_decision
    ON fraud_alerts (decision);

CREATE INDEX IF NOT EXISTS idx_alerts_created_at
    ON fraud_alerts (created_at DESC);
"""


async def init() -> None:
    print(f"Connecting to Postgres: {POSTGRES_DSN.split('@')[-1]}...")
    conn = await asyncpg.connect(POSTGRES_DSN)
    try:
        await conn.execute(DDL)
        print("Schema applied successfully.")

        # Report row counts
        tables = ["predictions", "fraud_alerts"]
        for t in tables:
            n = await conn.fetchval(f"SELECT COUNT(*) FROM {t}")
            print(f"  {t}: {n:,} rows")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(init())
