-- ATLAS-X Monitoring Schema
-- Includes original tables + enterprise feature tables

-- ── Original monitoring table ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS transaction_audit (
  transaction_id BIGINT PRIMARY KEY,
  fraud_prob DOUBLE PRECISION,
  decision TEXT,
  segment TEXT,
  dqn_action TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Feature 1: Event-Sourced Memory ──────────────────────────────────────────
-- Full audit trail for every transaction event (predict, flag, review, etc.)
CREATE TABLE IF NOT EXISTS transaction_events (
  id             SERIAL PRIMARY KEY,
  transaction_id TEXT        NOT NULL,
  event_type     TEXT        NOT NULL,  -- 'predicted', 'flagged', 'reviewed', 'blocked'
  data           JSONB       NOT NULL DEFAULT '{}',
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_txn_events_txn_id
  ON transaction_events (transaction_id);

CREATE INDEX IF NOT EXISTS idx_txn_events_created_at
  ON transaction_events (created_at DESC);

-- ── Predictions table (used by streaming consumer + event sourcing) ───────────
CREATE TABLE IF NOT EXISTS predictions (
  id                SERIAL PRIMARY KEY,
  transaction_id    TEXT        UNIQUE NOT NULL,
  fraud_probability FLOAT,
  graph_risk_score  FLOAT,
  decision          TEXT,
  customer_segment  TEXT,
  latency_ms        FLOAT,
  created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_decision
  ON predictions (decision);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at
  ON predictions (created_at DESC);

-- ── Fraud alerts table ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fraud_alerts (
  id                SERIAL PRIMARY KEY,
  transaction_id    TEXT        UNIQUE NOT NULL,
  fraud_probability FLOAT,
  graph_risk_score  FLOAT,
  decision          TEXT,
  customer_segment  TEXT,
  is_fraud_label    BOOL,
  created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ── Feature 6: pgvector transaction embeddings ────────────────────────────────
-- Requires pgvector extension: CREATE EXTENSION IF NOT EXISTS vector;
-- This table is created separately in the pgvector init script to handle
-- the case where pgvector is not installed.
-- See: src/api/main.py  _init_pgvector()
