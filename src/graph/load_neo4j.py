"""
ATLAS-X Neo4j Graph Loader
──────────────────────────
Step 1 (always):  export CSVs to data/neo4j/
Step 2 (--load):  apply schema constraints then bulk-load via LOAD CSV

Nodes
  Card    – card1-card2-card3 composite key   (100% coverage)
  Email   – P_emaildomain                     (84%)
  Address – addr1 billing region              (89%)
  Device  – DeviceType-DeviceInfo             (24%)

Relationships
  (Card)-[:USED_DEVICE]->(Device)
  (Card)-[:HAS_EMAIL]->(Email)
  (Card)-[:BILLING_ADDR]->(Address)

Usage:
  # Export CSVs only
  python -m src.graph.load_neo4j

  # Export + load into Neo4j
  python -m src.graph.load_neo4j --load

  # Use full dataset instead of 20% holdout
  python -m src.graph.load_neo4j --all-rows --load

Env vars (optional):
  NEO4J_URI       bolt://localhost:7687
  NEO4J_USER      neo4j
  NEO4J_PASSWORD  password123
"""
import argparse
import csv
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.optimization.threshold_optimizer import compute_customer_segments

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "data/processed/train_full_features.parquet"
OUT_DIR = REPO / "data/neo4j"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dataframe(use_all_rows: bool) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)

    if not use_all_rows:
        # Reproduce the exact 20% holdout used for model validation.
        segment_s, _ = compute_customer_segments(df)
        df = df.copy()
        df["customer_segment"] = segment_s
        y = df["isFraud"].astype(int)
        _, df = train_test_split(df, test_size=0.2, random_state=42, stratify=y)
        df = df.reset_index(drop=True)

    return df


# ── Node / relationship builders ─────────────────────────────────────────────

def _str(val) -> str:
    s = str(val) if pd.notna(val) else ""
    return s.strip()


def build_node_tables(df: pd.DataFrame):
    """Return (cards, devices, emails, addresses, rels) as lists of dicts."""

    # ── Card IDs ──────────────────────────────────────────────────────────────
    df = df.copy()
    df["card_id"] = (
        df["card1"].astype(str).str.strip() + "-" +
        df["card2"].fillna("NA").astype(str).str.strip() + "-" +
        df["card3"].fillna("NA").astype(str).str.strip()
    )

    # ── Device IDs ────────────────────────────────────────────────────────────
    has_device = df["DeviceType"].notna() & df["DeviceInfo"].notna()
    df["device_id"] = ""
    df.loc[has_device, "device_id"] = (
        df.loc[has_device, "DeviceType"].astype(object).astype(str).str.strip() + "-" +
        df.loc[has_device, "DeviceInfo"].astype(object).astype(str).str.strip()
    )

    # ── Email IDs ─────────────────────────────────────────────────────────────
    df["email_id"] = df["P_emaildomain"].astype(object).fillna("").astype(str).str.strip().str.lower()

    # ── Address IDs ───────────────────────────────────────────────────────────
    df["addr_id"] = df["addr1"].astype(object).fillna("").astype(str).str.strip()

    # ── Per-node fraud stats ──────────────────────────────────────────────────
    def fraud_stats(group_col: str, filter_nonempty: bool = True):
        sub = df[df[group_col] != ""] if filter_nonempty else df
        g = sub.groupby(group_col)["isFraud"]
        return (
            g.count().rename("total_txn"),
            g.sum().rename("fraud_txn"),
            g.mean().rename("fraud_rate"),
        )

    card_total, card_fraud, card_rate = fraud_stats("card_id", filter_nonempty=False)
    dev_total,  dev_fraud,  dev_rate  = fraud_stats("device_id")
    em_total,   em_fraud,   em_rate   = fraud_stats("email_id")
    addr_total, addr_fraud, addr_rate = fraud_stats("addr_id")

    # ── Card aggregate props ──────────────────────────────────────────────────
    sub_card = df[df["card_id"] != ""].copy()
    sub_card["card4_s"] = sub_card["card4"].astype(object).fillna("").astype(str)
    sub_card["card6_s"] = sub_card["card6"].astype(object).fillna("").astype(str)
    card_extra = (
        sub_card.groupby("card_id")
        .agg(
            avg_amount=("TransactionAmt", "mean"),
            card4=("card4_s", "first"),
            card6=("card6_s", "first"),
        )
    )

    cards = []
    for cid in card_total.index:
        row = {
            "card_id": cid,
            "total_txn": int(card_total[cid]),
            "fraud_txn": int(card_fraud[cid]),
            "fraud_rate": round(float(card_rate[cid]), 6),
            "avg_amount": round(float(card_extra.loc[cid, "avg_amount"]) if cid in card_extra.index else 0.0, 2),
            "card4": _str(card_extra.loc[cid, "card4"]) if cid in card_extra.index else "",
            "card6": _str(card_extra.loc[cid, "card6"]) if cid in card_extra.index else "",
        }
        cards.append(row)

    # ── Device nodes ─────────────────────────────────────────────────────────
    sub_dev = df[df["device_id"] != ""].copy()
    sub_dev["DeviceType_s"] = sub_dev["DeviceType"].astype(object).fillna("").astype(str)
    sub_dev["DeviceInfo_s"] = sub_dev["DeviceInfo"].astype(object).fillna("").astype(str)
    dev_extra = (
        sub_dev.groupby("device_id")
        .agg(
            device_type=("DeviceType_s", "first"),
            device_info=("DeviceInfo_s", "first"),
        )
    )
    devices = []
    for did in dev_total.index:
        row = {
            "device_id": did,
            "total_txn": int(dev_total[did]),
            "fraud_txn": int(dev_fraud[did]),
            "fraud_rate": round(float(dev_rate[did]), 6),
            "device_type": _str(dev_extra.loc[did, "device_type"]) if did in dev_extra.index else "",
            "device_info": _str(dev_extra.loc[did, "device_info"]) if did in dev_extra.index else "",
        }
        devices.append(row)

    # ── Email nodes ───────────────────────────────────────────────────────────
    emails = []
    for eid in em_total.index:
        emails.append({
            "email_id": eid,
            "email_domain": eid,
            "total_txn": int(em_total[eid]),
            "fraud_txn": int(em_fraud[eid]),
            "fraud_rate": round(float(em_rate[eid]), 6),
        })

    # ── Address nodes ─────────────────────────────────────────────────────────
    sub_addr = df[df["addr_id"] != ""].copy()
    sub_addr["addr2_s"] = sub_addr["addr2"].astype(object).fillna("").astype(str)
    addr_extra = (
        sub_addr.groupby("addr_id")
        .agg(addr2=("addr2_s", "first"))
    )
    addresses = []
    for aid in addr_total.index:
        addresses.append({
            "addr_id": aid,
            "addr1": aid,
            "addr2": _str(addr_extra.loc[aid, "addr2"]) if aid in addr_extra.index else "",
            "total_txn": int(addr_total[aid]),
            "fraud_txn": int(addr_fraud[aid]),
            "fraud_rate": round(float(addr_rate[aid]), 6),
        })

    # ── Relationships ─────────────────────────────────────────────────────────
    rel_cols = ["TransactionID", "card_id", "device_id", "email_id", "addr_id",
                "isFraud", "TransactionAmt", "Transaction_Hour"]
    rels = []
    for _, r in df[rel_cols].iterrows():
        rels.append({
            "transaction_id": int(r["TransactionID"]),
            "card_id": r["card_id"],
            "device_id": r["device_id"],
            "email_id": r["email_id"],
            "addr_id": r["addr_id"],
            "is_fraud": str(r["isFraud"] == 1).lower(),
            "amount": round(float(r["TransactionAmt"]), 2),
            "transaction_hour": int(r["Transaction_Hour"]),
        })

    return cards, devices, emails, addresses, rels


# ── CSV export ────────────────────────────────────────────────────────────────

def write_csv(rows: list, path: Path) -> None:
    if not rows:
        print(f"  [skip] {path.name} — no rows")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows):,} rows → {path}")


def export_csvs(df: pd.DataFrame) -> tuple:
    print("Building node and relationship tables...")
    cards, devices, emails, addresses, rels = build_node_tables(df)

    print("Writing CSV files...")
    write_csv(cards,     OUT_DIR / "cards.csv")
    write_csv(devices,   OUT_DIR / "devices.csv")
    write_csv(emails,    OUT_DIR / "emails.csv")
    write_csv(addresses, OUT_DIR / "addresses.csv")
    write_csv(rels,      OUT_DIR / "relationships.csv")

    return cards, devices, emails, addresses, rels


# ── Neo4j loader ──────────────────────────────────────────────────────────────

def _apply_schema(driver) -> None:
    schema_path = Path(__file__).parent / "schema.cypher"
    statements = [
        s.strip() for s in schema_path.read_text().split(";")
        if s.strip() and not s.strip().startswith("//")
    ]
    with driver.session() as session:
        for stmt in statements:
            try:
                session.run(stmt)
            except Exception as e:
                print(f"  [schema warn] {e}")


def _load_csv_batch(driver, cypher: str, rows: list, batch_size: int = 5000) -> None:
    with driver.session() as session:
        for i in range(0, len(rows), batch_size):
            session.run(cypher, rows=rows[i : i + batch_size])


def load_into_neo4j(cards, devices, emails, addresses, rels) -> None:
    from neo4j import GraphDatabase  # deferred so CSV-only mode has no hard dep

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        print("Applying schema constraints and indexes...")
        _apply_schema(driver)

        print(f"Loading {len(cards):,} Card nodes...")
        _load_csv_batch(driver, """
            UNWIND $rows AS row
            MERGE (c:Card {card_id: row.card_id})
            SET c.total_txn   = toInteger(row.total_txn),
                c.fraud_txn   = toInteger(row.fraud_txn),
                c.fraud_rate  = toFloat(row.fraud_rate),
                c.avg_amount  = toFloat(row.avg_amount),
                c.card4       = row.card4,
                c.card6       = row.card6
        """, cards)

        print(f"Loading {len(devices):,} Device nodes...")
        _load_csv_batch(driver, """
            UNWIND $rows AS row
            MERGE (d:Device {device_id: row.device_id})
            SET d.device_type = row.device_type,
                d.device_info = row.device_info,
                d.total_txn   = toInteger(row.total_txn),
                d.fraud_txn   = toInteger(row.fraud_txn),
                d.fraud_rate  = toFloat(row.fraud_rate)
        """, devices)

        print(f"Loading {len(emails):,} Email nodes...")
        _load_csv_batch(driver, """
            UNWIND $rows AS row
            MERGE (e:Email {email_id: row.email_id})
            SET e.email_domain = row.email_domain,
                e.total_txn    = toInteger(row.total_txn),
                e.fraud_txn    = toInteger(row.fraud_txn),
                e.fraud_rate   = toFloat(row.fraud_rate)
        """, emails)

        print(f"Loading {len(addresses):,} Address nodes...")
        _load_csv_batch(driver, """
            UNWIND $rows AS row
            MERGE (a:Address {addr_id: row.addr_id})
            SET a.addr1       = row.addr1,
                a.addr2       = row.addr2,
                a.total_txn   = toInteger(row.total_txn),
                a.fraud_txn   = toInteger(row.fraud_txn),
                a.fraud_rate  = toFloat(row.fraud_rate)
        """, addresses)

        print(f"Loading {len(rels):,} relationships...")
        _load_csv_batch(driver, """
            UNWIND $rows AS row
            MATCH (c:Card {card_id: row.card_id})

            // Device relationship (only when device_id is non-empty)
            FOREACH (_ IN CASE WHEN row.device_id <> '' THEN [1] ELSE [] END |
                MERGE (d:Device {device_id: row.device_id})
                MERGE (c)-[:USED_DEVICE]->(d)
            )

            // Email relationship
            FOREACH (_ IN CASE WHEN row.email_id <> '' THEN [1] ELSE [] END |
                MERGE (e:Email {email_id: row.email_id})
                MERGE (c)-[:HAS_EMAIL]->(e)
            )

            // Address relationship
            FOREACH (_ IN CASE WHEN row.addr_id <> '' THEN [1] ELSE [] END |
                MERGE (a:Address {addr_id: row.addr_id})
                MERGE (c)-[:BILLING_ADDR]->(a)
            )
        """, rels)

        print("Neo4j load complete.")
    finally:
        driver.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS-X Neo4j graph loader")
    parser.add_argument(
        "--load", action="store_true",
        help="After exporting CSVs, load directly into Neo4j via the driver"
    )
    parser.add_argument(
        "--all-rows", action="store_true",
        help="Use full dataset (590k rows) instead of 20%% holdout (120k rows)"
    )
    args = parser.parse_args()

    print(f"Loading {'full dataset' if args.all_rows else '20% holdout'} from parquet...")
    df = load_dataframe(use_all_rows=args.all_rows)
    print(f"  {len(df):,} transactions, {int(df['isFraud'].sum()):,} fraud")

    cards, devices, emails, addresses, rels = export_csvs(df)

    print(f"\nGraph summary:")
    print(f"  Card nodes:    {len(cards):,}")
    print(f"  Device nodes:  {len(devices):,}")
    print(f"  Email nodes:   {len(emails):,}")
    print(f"  Address nodes: {len(addresses):,}")
    print(f"  Relationships: {len(rels):,} transaction rows")

    if args.load:
        print(f"\nConnecting to Neo4j at {NEO4J_URI}...")
        load_into_neo4j(cards, devices, emails, addresses, rels)
    else:
        print(f"\nCSVs written to {OUT_DIR}/")
        print("Run with --load to ingest into Neo4j, or use data/neo4j/import.cypher manually.")


if __name__ == "__main__":
    main()
