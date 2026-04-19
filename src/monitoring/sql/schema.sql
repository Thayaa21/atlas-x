-- ATLAS-X Monitoring Schema

CREATE TABLE IF NOT EXISTS transaction_audit (
  transaction_id BIGINT PRIMARY KEY,
  fraud_prob DOUBLE PRECISION,
  decision TEXT,
  segment TEXT,
  dqn_action TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

