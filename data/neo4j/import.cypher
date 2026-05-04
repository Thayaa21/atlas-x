// ATLAS-X Neo4j Bulk Import
// ─────────────────────────────────────────────────────────────────────────────
// Prerequisites:
//   1. Run: python -m src.graph.load_neo4j    (generates the CSV files)
//   2. Copy the CSVs into Neo4j's import directory:
//        cp data/neo4j/*.csv <NEO4J_HOME>/import/
//      Or set dbms.directories.import=<absolute path to data/neo4j/> in neo4j.conf
//   3. Paste these statements into Neo4j Browser or run via cypher-shell:
//        cypher-shell -u neo4j -p password123 < data/neo4j/import.cypher
//
// Nodes created: Card, Device, Email, Address
// Relationships: USED_DEVICE, HAS_EMAIL, BILLING_ADDR
// ─────────────────────────────────────────────────────────────────────────────

// ── Schema ────────────────────────────────────────────────────────────────────

CREATE CONSTRAINT card_id_unique IF NOT EXISTS
FOR (c:Card) REQUIRE c.card_id IS UNIQUE;

CREATE CONSTRAINT device_id_unique IF NOT EXISTS
FOR (d:Device) REQUIRE d.device_id IS UNIQUE;

CREATE CONSTRAINT email_id_unique IF NOT EXISTS
FOR (e:Email) REQUIRE e.email_id IS UNIQUE;

CREATE CONSTRAINT addr_id_unique IF NOT EXISTS
FOR (a:Address) REQUIRE a.addr_id IS UNIQUE;

CREATE INDEX card_fraud_rate_idx IF NOT EXISTS FOR (c:Card) ON (c.fraud_rate);
CREATE INDEX card_pagerank_idx   IF NOT EXISTS FOR (c:Card) ON (c.pagerank);
CREATE INDEX card_community_idx  IF NOT EXISTS FOR (c:Card) ON (c.communityId);

// ── Card Nodes ────────────────────────────────────────────────────────────────

LOAD CSV WITH HEADERS FROM 'file:///cards.csv' AS row
MERGE (c:Card {card_id: row.card_id})
SET c.total_txn  = toInteger(row.total_txn),
    c.fraud_txn  = toInteger(row.fraud_txn),
    c.fraud_rate = toFloat(row.fraud_rate),
    c.avg_amount = toFloat(row.avg_amount),
    c.card4      = row.card4,
    c.card6      = row.card6;

// ── Device Nodes ──────────────────────────────────────────────────────────────

LOAD CSV WITH HEADERS FROM 'file:///devices.csv' AS row
MERGE (d:Device {device_id: row.device_id})
SET d.device_type = row.device_type,
    d.device_info = row.device_info,
    d.total_txn   = toInteger(row.total_txn),
    d.fraud_txn   = toInteger(row.fraud_txn),
    d.fraud_rate  = toFloat(row.fraud_rate);

// ── Email Nodes ───────────────────────────────────────────────────────────────

LOAD CSV WITH HEADERS FROM 'file:///emails.csv' AS row
MERGE (e:Email {email_id: row.email_id})
SET e.email_domain = row.email_domain,
    e.total_txn    = toInteger(row.total_txn),
    e.fraud_txn    = toInteger(row.fraud_txn),
    e.fraud_rate   = toFloat(row.fraud_rate);

// ── Address Nodes ─────────────────────────────────────────────────────────────

LOAD CSV WITH HEADERS FROM 'file:///addresses.csv' AS row
MERGE (a:Address {addr_id: row.addr_id})
SET a.addr1      = row.addr1,
    a.addr2      = row.addr2,
    a.total_txn  = toInteger(row.total_txn),
    a.fraud_txn  = toInteger(row.fraud_txn),
    a.fraud_rate = toFloat(row.fraud_rate);

// ── Relationships ─────────────────────────────────────────────────────────────
// One row per transaction. device_id / email_id / addr_id may be empty strings
// (NULL in the graph sense) — the FOREACH guards skip those safely.

LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
MATCH (c:Card {card_id: row.card_id})
FOREACH (_ IN CASE WHEN row.device_id <> '' THEN [1] ELSE [] END |
    MERGE (d:Device {device_id: row.device_id})
    MERGE (c)-[:USED_DEVICE]->(d)
)
FOREACH (_ IN CASE WHEN row.email_id <> '' THEN [1] ELSE [] END |
    MERGE (e:Email {email_id: row.email_id})
    MERGE (c)-[:HAS_EMAIL]->(e)
)
FOREACH (_ IN CASE WHEN row.addr_id <> '' THEN [1] ELSE [] END |
    MERGE (a:Address {addr_id: row.addr_id})
    MERGE (c)-[:BILLING_ADDR]->(a)
);

// ── Verify ────────────────────────────────────────────────────────────────────

MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC;

MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS count ORDER BY count DESC;

// ── GDS: PageRank + Louvain (requires Neo4j GDS plugin) ──────────────────────
// Run these separately after the import above is confirmed.

// Project the graph
CALL gds.graph.project(
  'atlasx_fraud_graph',
  ['Card', 'Device', 'Email', 'Address'],
  {
    USED_DEVICE:  {type: 'USED_DEVICE',  orientation: 'UNDIRECTED'},
    HAS_EMAIL:    {type: 'HAS_EMAIL',    orientation: 'UNDIRECTED'},
    BILLING_ADDR: {type: 'BILLING_ADDR', orientation: 'UNDIRECTED'}
  }
);

// PageRank — writes Card.pagerank
CALL gds.pageRank.write('atlasx_fraud_graph', {
  maxIterations: 30,
  dampingFactor: 0.85,
  writeProperty: 'pagerank'
});

// Louvain community detection — writes Card.communityId
CALL gds.louvain.write('atlasx_fraud_graph', {
  writeProperty: 'communityId',
  relationshipTypes: ['USED_DEVICE', 'HAS_EMAIL', 'BILLING_ADDR']
});

// Community fraud risk — top 20 highest-risk rings
MATCH (c:Card)
WHERE c.communityId IS NOT NULL
WITH c.communityId AS ring_id,
     count(c)                          AS ring_size,
     avg(c.fraud_rate)                 AS avg_fraud_rate,
     sum(c.fraud_txn)                  AS total_fraud_txn,
     sum(c.total_txn)                  AS total_txn
WHERE ring_size >= 3
RETURN ring_id, ring_size, round(avg_fraud_rate, 4) AS avg_fraud_rate,
       total_fraud_txn, total_txn
ORDER BY avg_fraud_rate DESC
LIMIT 20;
