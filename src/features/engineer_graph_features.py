"""
ATLAS-X Graph Feature Engineering
───────────────────────────────────
Reads the Neo4j CSVs already exported to data/neo4j/ and computes
card-level graph topology features:

  device_fraud_rate          – fraud rate of the most suspicious device
                               this card used (proxy for neighbourhood risk)
  device_card_velocity       – distinct cards on same device (crowding signal)
  connected_fraud_cards      – other CONFIRMED fraud cards sharing a device
                               with this card (direct ring signal)
  email_fraud_rate           – fraud rate of this card's email domain
  address_fraud_rate         – fraud rate of this card's billing address
  graph_risk_score           – weighted composite (0-1)

These 6 features are then joined on card_id into the full feature parquet
so that the v4 model can learn from graph topology during training instead
of applying a post-hoc rule (which caused FP explosion at evaluation time).

NOTE on leakage:
  The CSVs are built from the 20% holdout by default (see load_neo4j.py).
  Cards that only appear in the 80% training split receive 0 for all graph
  features.  This is conservative — a real production pipeline would
  regenerate graph stats from all historical data.  For this comparative
  evaluation the approach is consistent and unbiased across v3/v4.

Usage:
    python -m src.features.engineer_graph_features
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO      = Path(__file__).resolve().parents[2]
NEO4J_DIR = REPO / "data/neo4j"
DATA_IN   = REPO / "data/processed/train_full_features.parquet"
DATA_OUT  = REPO / "data/processed/train_with_graph_features.parquet"


# ── Load CSVs ─────────────────────────────────────────────────────────────────

def _load_csvs():
    cards     = pd.read_csv(NEO4J_DIR / "cards.csv")
    devices   = pd.read_csv(NEO4J_DIR / "devices.csv")
    emails    = pd.read_csv(NEO4J_DIR / "emails.csv")
    addresses = pd.read_csv(NEO4J_DIR / "addresses.csv")
    rels      = pd.read_csv(
        NEO4J_DIR / "relationships.csv",
        dtype={"device_id": str, "email_id": str, "addr_id": str},
        keep_default_na=False,
    )
    # Normalise empty strings → NaN for easy filtering
    for col in ["device_id", "email_id", "addr_id"]:
        rels[col] = rels[col].replace("", np.nan)
    return cards, devices, emails, addresses, rels


# ── Device features ───────────────────────────────────────────────────────────

def _device_features(cards: pd.DataFrame, devices: pd.DataFrame,
                     rels: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a card-level DataFrame with:
      device_fraud_rate    – max fraud_rate across devices used by this card
      device_card_velocity – max unique-card-count across those devices
      connected_fraud_cards– fraud cards sharing a device (excluding self)
    """
    # ── (card, device) pairs ──────────────────────────────────────────────────
    cd = (
        rels.dropna(subset=["device_id"])[["card_id", "device_id"]]
        .drop_duplicates()
    )

    # ── Per-device: how many unique cards? ───────────────────────────────────
    dev_card_cnt = (
        cd.groupby("device_id")["card_id"].nunique()
        .reset_index(name="device_card_velocity")
    )

    # ── Per-device: how many are confirmed fraud cards? ───────────────────────
    fraud_card_set = set(cards.loc[cards["fraud_txn"] > 0, "card_id"])
    cd_fraud = cd.copy()
    cd_fraud["is_fraud_card"] = cd_fraud["card_id"].isin(fraud_card_set).astype(int)
    dev_fraud_cnt = (
        cd_fraud.groupby("device_id")["is_fraud_card"].sum()
        .reset_index(name="device_fraud_card_count")
    )

    # ── Join device stats back to (card, device) ──────────────────────────────
    dev_stats = (
        devices[["device_id", "fraud_rate"]]
        .rename(columns={"fraud_rate": "dev_fraud_rate"})
        .merge(dev_card_cnt,  on="device_id", how="left")
        .merge(dev_fraud_cnt, on="device_id", how="left")
        .fillna(0)
    )
    cd2 = cd.merge(dev_stats, on="device_id", how="left")

    # ── Aggregate per card: take max signal across all devices used ───────────
    agg = (
        cd2.groupby("card_id")
        .agg(
            device_fraud_rate   =("dev_fraud_rate",          "max"),
            device_card_velocity=("device_card_velocity",    "max"),
            raw_fraud_cnt       =("device_fraud_card_count", "max"),
        )
        .reset_index()
    )

    # Subtract self from fraud count (a card shouldn't count itself)
    card_is_fraud = (
        cards[["card_id"]]
        .assign(self_fraud=lambda df: df["card_id"].isin(fraud_card_set).astype(int))
    )
    agg = agg.merge(card_is_fraud, on="card_id", how="left")
    agg["connected_fraud_cards"] = (agg["raw_fraud_cnt"] - agg["self_fraud"]).clip(lower=0)
    return agg.drop(columns=["raw_fraud_cnt", "self_fraud"])


# ── Email features ────────────────────────────────────────────────────────────

def _email_features(emails: pd.DataFrame, rels: pd.DataFrame) -> pd.DataFrame:
    ce = (
        rels.dropna(subset=["email_id"])[["card_id", "email_id"]]
        .drop_duplicates()
        .merge(
            emails[["email_id", "fraud_rate"]].rename(columns={"fraud_rate": "email_fraud_rate"}),
            on="email_id", how="left",
        )
    )
    return (
        ce.groupby("card_id")["email_fraud_rate"].max()
        .reset_index()
    )


# ── Address features ──────────────────────────────────────────────────────────

def _address_features(addresses: pd.DataFrame, rels: pd.DataFrame) -> pd.DataFrame:
    # addr_id in addresses.csv is numeric (addr1 values); normalise both to str
    addr_lookup = addresses[["addr_id", "fraud_rate"]].copy()
    addr_lookup["addr_id"] = addr_lookup["addr_id"].astype(str).str.strip()
    ca = (
        rels.dropna(subset=["addr_id"])[["card_id", "addr_id"]]
        .drop_duplicates()
        .assign(addr_id=lambda df: df["addr_id"].astype(str).str.strip())
        .merge(
            addr_lookup.rename(columns={"fraud_rate": "address_fraud_rate"}),
            on="addr_id", how="left",
        )
    )
    return (
        ca.groupby("card_id")["address_fraud_rate"].max()
        .reset_index()
    )


# ── Assemble card-level graph feature table ───────────────────────────────────

def build_graph_features() -> pd.DataFrame:
    print("Loading Neo4j CSVs...")
    cards, devices, emails, addresses, rels = _load_csvs()
    print(f"  cards={len(cards):,}  devices={len(devices):,}  "
          f"emails={len(emails):,}  addresses={len(addresses):,}  "
          f"rels={len(rels):,}")

    print("Computing device features...")
    dev_feat  = _device_features(cards, devices, rels)
    print(f"  {len(dev_feat):,} cards with device linkage")

    print("Computing email features...")
    em_feat   = _email_features(emails, rels)

    print("Computing address features...")
    addr_feat = _address_features(addresses, rels)

    # Merge onto the full card list so every card gets a row (0 = no link)
    gf = cards[["card_id"]].copy()
    gf = gf.merge(dev_feat,  on="card_id", how="left")
    gf = gf.merge(em_feat,   on="card_id", how="left")
    gf = gf.merge(addr_feat, on="card_id", how="left")

    graph_cols = [
        "device_fraud_rate", "device_card_velocity", "connected_fraud_cards",
        "email_fraud_rate", "address_fraud_rate",
    ]
    gf[graph_cols] = gf[graph_cols].fillna(0.0)

    # Composite risk score: device signal dominates (more specific)
    gf["graph_risk_score"] = (
        0.5 * gf["device_fraud_rate"] +
        0.3 * gf["email_fraud_rate"] +
        0.2 * gf["address_fraud_rate"]
    ).clip(0, 1)

    return gf


# ── Join into full feature parquet ────────────────────────────────────────────

def engineer(save: bool = True) -> pd.DataFrame:
    gf = build_graph_features()

    print(f"\nLoading base features from {DATA_IN.name}...")
    df = pd.read_parquet(DATA_IN)
    print(f"  {len(df):,} transactions, {int(df['isFraud'].sum()):,} fraud")
    print(f"  Original feature count: {df.shape[1]}")

    # Reconstruct card_id for join (same logic as load_neo4j.py)
    df["card_id"] = (
        df["card1"].astype(str).str.strip() + "-" +
        df["card2"].fillna("NA").astype(str).str.strip() + "-" +
        df["card3"].fillna("NA").astype(str).str.strip()
    )

    df = df.merge(gf.drop(columns=["card_id"], errors="ignore").assign(card_id=gf["card_id"]),
                  on="card_id", how="left")

    new_cols = [
        "device_fraud_rate", "device_card_velocity", "connected_fraud_cards",
        "email_fraud_rate", "address_fraud_rate", "graph_risk_score",
    ]
    df[new_cols] = df[new_cols].fillna(0.0)

    # Drop the helper card_id column (not used as a model feature)
    df = df.drop(columns=["card_id"])

    print(f"  Updated  feature count: {df.shape[1]}  (+{df.shape[1] - (df.shape[1] - len(new_cols))} graph cols)")

    # ── Coverage stats ────────────────────────────────────────────────────────
    n_with_device = (df["device_fraud_rate"] > 0).sum()
    n_in_ring     = (df["connected_fraud_cards"] >= 2).sum()
    print(f"\nGraph feature coverage:")
    print(f"  Transactions with device link : {n_with_device:,} / {len(df):,} ({n_with_device/len(df):.1%})")
    print(f"  Transactions in fraud ring (≥2): {n_in_ring:,} / {len(df):,} ({n_in_ring/len(df):.1%})")

    if save:
        DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(DATA_OUT, index=False)
        print(f"\nSaved → {DATA_OUT}")

    return df


if __name__ == "__main__":
    engineer()
