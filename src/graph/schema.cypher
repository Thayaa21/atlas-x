// ATLAS-X Fraud Graph Schema
// Node properties:
// - Card:   card_id
// - Device: device_id
// - IP:     ip_id
// - Email:  email_id
//
// Relationship properties are optional; risk is stored on nodes.

CREATE CONSTRAINT card_id_unique IF NOT EXISTS
FOR (c:Card)
REQUIRE c.card_id IS UNIQUE;

CREATE CONSTRAINT device_id_unique IF NOT EXISTS
FOR (d:Device)
REQUIRE d.device_id IS UNIQUE;

CREATE CONSTRAINT ip_id_unique IF NOT EXISTS
FOR (i:IP)
REQUIRE i.ip_id IS UNIQUE;

CREATE CONSTRAINT email_id_unique IF NOT EXISTS
FOR (e:Email)
REQUIRE e.email_id IS UNIQUE;

CREATE CONSTRAINT addr_id_unique IF NOT EXISTS
FOR (a:Address)
REQUIRE a.addr_id IS UNIQUE;

// Indexes to speed up lookups
CREATE INDEX card_fraud_rate_idx IF NOT EXISTS FOR (c:Card) ON (c.fraud_rate);
CREATE INDEX card_pagerank_idx   IF NOT EXISTS FOR (c:Card) ON (c.pagerank);
CREATE INDEX card_community_idx  IF NOT EXISTS FOR (c:Card) ON (c.communityId);

