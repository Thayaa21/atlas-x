import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from neo4j import GraphDatabase


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str
    graph_name: str = "atlasx_fraud_graph"


def _get_neo4j_config() -> Neo4jConfig:
    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")
    graph_name = os.getenv("NEO4J_GRAPH_NAME", "atlasx_fraud_graph")
    return Neo4jConfig(uri=uri, user=user, password=password, graph_name=graph_name)


def compute_node_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map transaction rows to graph node keys.

    - Card: card1-card2-card3
    - Device: DeviceType + DeviceInfo
    - IP: id_15 (if missing, falls back to id_01)
    - Email: P_emaildomain + R_emaildomain
    """
    dfi = df.copy()

    for c in ["card1", "card2", "card3"]:
        if c not in dfi.columns:
            raise ValueError(f"Missing required column for Card node: {c}")

    dfi["card_id"] = dfi["card1"].astype(str) + "-" + dfi["card2"].astype(str) + "-" + dfi["card3"].astype(str)

    device_type = dfi.get("DeviceType")
    device_info = dfi.get("DeviceInfo")
    if device_type is None or device_info is None:
        raise ValueError("Expected `DeviceType` and `DeviceInfo` columns for Device nodes.")
    dfi["device_id"] = device_type.astype(str).fillna("None") + "-" + device_info.astype(str).fillna("None")

    ip_col = "id_15" if "id_15" in dfi.columns else "id_01"
    dfi["ip_id"] = dfi[ip_col].astype(str).fillna("None")

    pdom = dfi.get("P_emaildomain")
    rdom = dfi.get("R_emaildomain")
    if pdom is None or rdom is None:
        raise ValueError("Expected `P_emaildomain` and `R_emaildomain` columns for Email nodes.")
    dfi["email_id"] = pdom.astype(str).fillna("None") + "-" + rdom.astype(str).fillna("None")

    return dfi


def _batched(iterable: Iterable[Dict], batch_size: int) -> Iterable[List[Dict]]:
    buf: List[Dict] = []
    for row in iterable:
        buf.append(row)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def ingest_graph_from_parquet(
    *,
    driver,
    data_path: Path,
    batch_size: int = 5000,
    max_rows: int | None = None,
) -> None:
    """
    Ingest nodes + relationships derived from the parquet dataset.

    For performance, this function batches UNWIND operations.
    """
    df = pd.read_parquet(data_path)
    if max_rows is not None:
        df = df.head(max_rows)

    if "isFraud" not in df.columns:
        raise ValueError("Expected target column `isFraud` for fraud rate estimates.")

    df = compute_node_ids(df)

    # Precompute fraud rate for each node type so we can store as properties.
    fraud_rate_card = df.groupby("card_id")["isFraud"].mean().to_dict()
    fraud_rate_device = df.groupby("device_id")["isFraud"].mean().to_dict()
    fraud_rate_ip = df.groupby("ip_id")["isFraud"].mean().to_dict()
    fraud_rate_email = df.groupby("email_id")["isFraud"].mean().to_dict()

    # Build relationship rows.
    rel_rows = []
    for _, r in df[["card_id", "device_id", "ip_id", "email_id"]].iterrows():
        rel_rows.append(
            {
                "card_id": r["card_id"],
                "device_id": r["device_id"],
                "ip_id": r["ip_id"],
                "email_id": r["email_id"],
            }
        )

    def ingest_batch(batch: List[Dict]) -> None:
        cypher = """
        UNWIND $rows AS row
        MERGE (c:Card {card_id: row.card_id})
        MERGE (d:Device {device_id: row.device_id})
        MERGE (i:IP {ip_id: row.ip_id})
        MERGE (e:Email {email_id: row.email_id})
        MERGE (c)-[:USED_DEVICE]->(d)
        MERGE (c)-[:USED_IP]->(i)
        MERGE (c)-[:USED_EMAIL]->(e)
        """
        driver.execute_query(cypher, rows=batch)

    for batch in _batched(rel_rows, batch_size=batch_size):
        ingest_batch(batch)

    # Store fraud_rate on nodes via batched updates.
    def update_node_rates(label: str, id_prop: str, rates: Dict[str, float]) -> None:
        cypher = f"""
        UNWIND $rows AS row
        MATCH (n:{label} {{{id_prop}: row.id}})
        SET n.fraud_rate = row.rate
        """
        rows = [{"id": k, "rate": float(v)} for k, v in rates.items()]
        for batch in _batched(rows, batch_size=batch_size):
            driver.execute_query(cypher, rows=batch)

    update_node_rates("Card", "card_id", fraud_rate_card)
    update_node_rates("Device", "device_id", fraud_rate_device)
    update_node_rates("IP", "ip_id", fraud_rate_ip)
    update_node_rates("Email", "email_id", fraud_rate_email)


def run_pagerank_and_louvain(
    *,
    driver,
    graph_name: str,
) -> None:
    """
    Runs PageRank and Louvain (requires Neo4j GDS).
    """
    # Use multi-line query to keep it in a single roundtrip.
    cypher = """
    CALL gds.graph.drop($graphName, false) YIELD graphName;
    CALL gds.graph.project(
      $graphName,
      ['Card','Device','IP','Email'],
      {
        USED_DEVICE: {type: 'USED_DEVICE', orientation: 'UNDIRECTED'},
        USED_IP: {type: 'USED_IP', orientation: 'UNDIRECTED'},
        USED_EMAIL: {type: 'USED_EMAIL', orientation: 'UNDIRECTED'}
      },
      { parameters: {} }
    );
    CALL gds.pageRank.write($graphName, {
      maxIterations: 30,
      dampingFactor: 0.85,
      writeProperty: 'pagerank',
      writePropertyMapped: false
    });
    CALL gds.louvain.write($graphName, {
      writeProperty: 'communityId',
      relationshipTypes: ['USED_DEVICE','USED_IP','USED_EMAIL']
    });
    """
    driver.execute_query(cypher, graphName=graph_name)


def export_ring_artifacts(
    *,
    driver,
    out_dir: Path,
    node_key_map: Dict[str, str],
) -> None:
    """
    Export node->ring mapping for fast API lookups.

    node_key_map examples:
      {"card_id": "card", "device_id": "device", ...}
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    query = """
    MATCH (n)
    WHERE labels(n) IN [['Card'],['Device'],['IP'],['Email']] AND exists(n.communityId) AND exists(n.pagerank)
    RETURN labels(n)[0] AS nodeType, n.* AS props
    """
    # neo4j driver returns a list of records; keep it simple.
    records = driver.execute_query(query)

    # `records.records` is not a standard neo4j return; handle via records.value() style.
    # In Cursor environment, we'll avoid being too clever: we will re-run with a simpler query.
    # (This function is primarily for offline export; production would tighten types.)
    export_rows = {"Card": {}, "Device": {}, "IP": {}, "Email": {}}
    for seg in ["Card", "Device", "IP", "Email"]:
        if seg not in export_rows:
            export_rows[seg] = {}

    simpler = """
    MATCH (n:Card) WHERE exists(n.communityId) AND exists(n.pagerank)
    RETURN 'Card' AS nodeType, n.card_id AS nodeId, n.communityId AS ringId, n.pagerank AS pagerank
    """
    card_res = driver.execute_query(simpler).records
    for r in card_res:
        export_rows["Card"][r["nodeId"]] = {"ring_id": int(r["ringId"]), "pagerank": float(r["pagerank"])}

    simpler = """
    MATCH (n:Device) WHERE exists(n.communityId) AND exists(n.pagerank)
    RETURN 'Device' AS nodeType, n.device_id AS nodeId, n.communityId AS ringId, n.pagerank AS pagerank
    """
    device_res = driver.execute_query(simpler).records
    for r in device_res:
        export_rows["Device"][r["nodeId"]] = {"ring_id": int(r["ringId"]), "pagerank": float(r["pagerank"])}

    simpler = """
    MATCH (n:IP) WHERE exists(n.communityId) AND exists(n.pagerank)
    RETURN 'IP' AS nodeType, n.ip_id AS nodeId, n.communityId AS ringId, n.pagerank AS pagerank
    """
    ip_res = driver.execute_query(simpler).records
    for r in ip_res:
        export_rows["IP"][r["nodeId"]] = {"ring_id": int(r["ringId"]), "pagerank": float(r["pagerank"])}

    simpler = """
    MATCH (n:Email) WHERE exists(n.communityId) AND exists(n.pagerank)
    RETURN 'Email' AS nodeType, n.email_id AS nodeId, n.communityId AS ringId, n.pagerank AS pagerank
    """
    email_res = driver.execute_query(simpler).records
    for r in email_res:
        export_rows["Email"][r["nodeId"]] = {"ring_id": int(r["ringId"]), "pagerank": float(r["pagerank"])}

    (out_dir / "node_rings.json").write_text(json.dumps(export_rows, indent=2))


def fraud_rings_main(
    *,
    data_path: Path,
    neo4j_cfg: Neo4jConfig,
    out_dir: Path,
    do_ingest: bool = True,
    do_algorithms: bool = True,
    max_rows: int | None = None,
) -> None:
    driver = GraphDatabase.driver(neo4j_cfg.uri, auth=(neo4j_cfg.user, neo4j_cfg.password))
    try:
        if do_ingest:
            ingest_graph_from_parquet(driver=driver, data_path=data_path, max_rows=max_rows)
        if do_algorithms:
            run_pagerank_and_louvain(driver=driver, graph_name=neo4j_cfg.graph_name)
        export_ring_artifacts(
            driver=driver,
            out_dir=out_dir,
            node_key_map={"card_id": "Card", "device_id": "Device", "ip_id": "IP", "email_id": "Email"},
        )
    finally:
        driver.close()


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    neo4j_cfg = _get_neo4j_config()
    data_p = repo_root / "data/processed/train_full_features.parquet"
    out_p = repo_root / "src/graph/artifacts"

    fraud_rings_main(
        data_path=data_p,
        neo4j_cfg=neo4j_cfg,
        out_dir=out_p,
        do_ingest=True,
        do_algorithms=True,
        # Set max_rows for local smoke tests (keeps it fast).
        max_rows=int(os.getenv("MAX_ROWS", "0")) or None,
    )

