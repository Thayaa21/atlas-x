// ATLAS-X Fraud Ring Algorithms (requires Neo4j Graph Data Science - GDS)
//
// Relationships:
//   (c:Card)-[:USED_DEVICE]->(d:Device)
//   (c:Card)-[:HAS_EMAIL]->(e:Email)
//   (c:Card)-[:BILLING_ADDR]->(a:Address)
//
// Stores on nodes:
//   Card.pagerank     (PageRank score)
//   Card.communityId  (Louvain community id)

// ── Graph projection ──────────────────────────────────────────────────────────

CALL gds.graph.drop($graphName, false) YIELD graphName;

CALL gds.graph.project(
  $graphName,
  ['Card', 'Device', 'Email', 'Address'],
  {
    USED_DEVICE:  {type: 'USED_DEVICE',  orientation: 'UNDIRECTED'},
    HAS_EMAIL:    {type: 'HAS_EMAIL',    orientation: 'UNDIRECTED'},
    BILLING_ADDR: {type: 'BILLING_ADDR', orientation: 'UNDIRECTED'}
  }
);

// ── PageRank ──────────────────────────────────────────────────────────────────

CALL gds.pageRank.write($graphName, {
  maxIterations: 30,
  dampingFactor: 0.85,
  writeProperty: 'pagerank'
});

// ── Louvain community detection ───────────────────────────────────────────────

CALL gds.louvain.write($graphName, {
  writeProperty: 'communityId',
  relationshipTypes: ['USED_DEVICE', 'HAS_EMAIL', 'BILLING_ADDR']
});

// ── Community fraud risk — top 20 rings ───────────────────────────────────────

MATCH (c:Card)
WHERE c.communityId IS NOT NULL
WITH c.communityId AS ring_id,
     count(c)          AS ring_size,
     avg(c.fraud_rate) AS avg_fraud_rate,
     sum(c.fraud_txn)  AS total_fraud_txn,
     sum(c.total_txn)  AS total_txn
WHERE ring_size >= 3
RETURN ring_id,
       ring_size,
       round(avg_fraud_rate, 4) AS avg_fraud_rate,
       total_fraud_txn,
       total_txn
ORDER BY avg_fraud_rate DESC
LIMIT 20;

// ── Lookup: rings for a given Card ───────────────────────────────────────────

MATCH (c:Card {card_id: $card_id})
OPTIONAL MATCH (c)-[:USED_DEVICE]->(d:Device)
OPTIONAL MATCH (c)-[:HAS_EMAIL]->(e:Email)
OPTIONAL MATCH (c)-[:BILLING_ADDR]->(a:Address)
RETURN c.card_id        AS card_id,
       c.fraud_rate     AS card_fraud_rate,
       c.communityId    AS ring_id,
       c.pagerank       AS pagerank,
       d.device_id      AS device_id,
       d.fraud_rate     AS device_fraud_rate,
       e.email_id       AS email_id,
       e.fraud_rate     AS email_fraud_rate,
       a.addr_id        AS addr_id,
       a.fraud_rate     AS addr_fraud_rate;

// ── High-risk ring members (ring_id supplied by API) ─────────────────────────

MATCH (c:Card {communityId: $ring_id})
RETURN c.card_id     AS card_id,
       c.fraud_rate  AS fraud_rate,
       c.pagerank    AS pagerank,
       c.total_txn   AS total_txn,
       c.fraud_txn   AS fraud_txn
ORDER BY c.pagerank DESC
LIMIT 50;
