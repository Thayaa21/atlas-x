"""
ATLAS-X Kafka Producer – Transaction Simulator
────────────────────────────────────────────────
Reads the 20% holdout from train_with_graph_features.parquet and publishes
each transaction to the 'transactions' Kafka topic in the format expected by
the /api/v1/predict endpoint.

Usage:
    python -m src.streaming.kafka_producer [--rate 100] [--count 1000]

Environment:
    KAFKA_BOOTSTRAP_SERVERS  default: localhost:29092
    KAFKA_TOPIC_TRANSACTIONS default: transactions
"""
import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from aiokafka import AIOKafkaProducer
from sklearn.model_selection import StratifiedShuffleSplit

REPO = Path(__file__).resolve().parents[2]
DATA_P = REPO / "data/processed/train_with_graph_features.parquet"

GRAPH_COLS = [
    "device_fraud_rate", "device_card_velocity", "connected_fraud_cards",
    "email_fraud_rate", "address_fraud_rate", "graph_risk_score",
]
DROP_COLS = ["isFraud", "TransactionID", "TransactionDT", "customer_segment"]


# ── Data prep ─────────────────────────────────────────────────────────────────

def _safe(v: Any) -> Any:
    """Convert numpy/NaN values to JSON-safe Python primitives."""
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def load_holdout() -> list[dict]:
    """
    Returns a list of message dicts ready for the /api/v1/predict endpoint.
    Uses the exact same 20% stratified split as model training.
    """
    from src.optimization.threshold_optimizer import compute_customer_segments

    print("Loading parquet...")
    df = pd.read_parquet(DATA_P)
    seg_s, _ = compute_customer_segments(df)
    df = df.copy()
    df["customer_segment"] = seg_s.values
    y = df["isFraud"].astype(int).to_numpy()

    # Feature matrix (mirrors train_v4)
    X = df.drop([c for c in DROP_COLS if c in df.columns], axis=1)
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].astype("object").fillna("None").astype("category")
    for col in GRAPH_COLS:
        if col in X.columns:
            X[col] = X[col].fillna(0.0)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(sss.split(X, y))

    X_test  = X.iloc[test_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    print(f"  {len(df_test):,} holdout rows  ({int(df_test['isFraud'].sum()):,} fraud)")

    messages = []
    for i in range(len(df_test)):
        row    = df_test.iloc[i]
        x_row  = X_test.iloc[i]
        card_id = (
            str(row["card1"]) + "-"
            + (str(int(row["card2"])) if pd.notna(row["card2"]) else "NA") + "-"
            + (str(int(row["card3"])) if pd.notna(row["card3"]) else "NA")
        )
        feat = {col: _safe(x_row[col]) for col in x_row.index}
        messages.append({
            "transaction_id": str(int(row["TransactionID"])),
            "card_id":        card_id,
            "amount":         float(row["TransactionAmt"]),
            "features":       feat,
            "is_fraud":       int(row["isFraud"]),   # ground-truth label (for monitoring)
        })
    return messages


# ── Async producer ────────────────────────────────────────────────────────────

async def produce(
    *,
    bootstrap: str = "localhost:29092",
    topic:     str = "transactions",
    messages:  list[dict],
    rate:      float = 100.0,
    count:     int   = 0,
) -> None:
    """
    Publish messages to Kafka at the given rate (transactions/sec).
    count=0 means publish all messages.
    """
    if count > 0:
        messages = messages[:count]

    interval = 1.0 / rate
    total    = len(messages)

    producer = AIOKafkaProducer(
        bootstrap_servers=bootstrap,
        acks="all",
        linger_ms=5,
        max_batch_size=131_072,
    )
    await producer.start()

    print(f"Publishing {total:,} transactions to '{topic}' at {rate:.0f} txns/sec...")
    t_start = time.perf_counter()
    sent = 0
    t_batch = t_start

    try:
        for i, msg in enumerate(messages):
            payload = json.dumps(msg, default=str).encode("utf-8")
            await producer.send_and_wait(topic, payload)
            sent += 1

            # Progress every 100 messages
            if sent % 100 == 0:
                elapsed = time.perf_counter() - t_start
                tps     = sent / elapsed
                pct     = sent / total * 100
                print(f"  [{pct:5.1f}%] {sent:>7,}/{total:,}  |  {tps:6.1f} txns/sec", flush=True)

            # Rate control: sleep to hit target rate
            expected_time = t_start + (i + 1) * interval
            sleep_for = expected_time - time.perf_counter()
            if sleep_for > 0.001:
                await asyncio.sleep(sleep_for)

    finally:
        await producer.stop()

    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {sent:,} messages in {elapsed:.1f}s  ({sent/elapsed:.1f} txns/sec)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    import os
    p = argparse.ArgumentParser(description="ATLAS-X Kafka producer")
    p.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"))
    p.add_argument("--topic",     default=os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions"))
    p.add_argument("--rate",      type=float, default=100.0, help="txns/sec")
    p.add_argument("--count",     type=int,   default=500,   help="txns per batch (0=all)")
    p.add_argument("--loop",      action="store_true",       help="Loop forever, sending batches every 10s")
    return p.parse_args()


def run_producer(bootstrap: str, topic: str, messages: list[dict],
                 rate: float, count: int) -> None:
    asyncio.run(produce(bootstrap=bootstrap, topic=topic,
                        messages=messages, rate=rate, count=count))


if __name__ == "__main__":
    args = _parse_args()
    msgs = load_holdout()

    if args.loop:
        print(f"🔁 Infinite mode: {args.count} txns/batch at {args.rate}/sec, 10s between batches")
        while True:
            run_producer(args.bootstrap, args.topic, msgs, args.rate, args.count)
            print("Batch complete. Sleeping 10s before next batch...")
            time.sleep(10)
    else:
        run_producer(args.bootstrap, args.topic, msgs, args.rate, args.count)
