import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from aiokafka import AIOKafkaProducer


def _get_env(name: str, default: str) -> str:
    v = os.getenv(name, default)
    if not v:
        raise RuntimeError(f"Missing env var {name}")
    return v


async def produce_transactions() -> None:
    kafka_bootstrap = _get_env("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
    topic = _get_env("KAFKA_TOPIC_TRANSACTIONS", "transactions")

    raw_dir = Path(_get_env("RAW_DATA_DIR", "data/raw"))
    txn_path = raw_dir / "train_transaction.csv"
    id_path = raw_dir / "train_identity.csv"

    if not txn_path.exists() or not id_path.exists():
        raise FileNotFoundError(
            f"Raw CSVs not found. Expected {txn_path} and {id_path}. "
            "Set RAW_DATA_DIR or place IEEE-CIS CSVs into data/raw/."
        )

    print("[KAFKA PRODUCER] Loading raw CSVs...")
    train_txn = pd.read_csv(txn_path)
    train_id = pd.read_csv(id_path)
    merged = pd.merge(train_txn, train_id, on="TransactionID", how="left")

    # Optional cap for local smoke tests
    max_rows = int(os.getenv("PRODUCER_MAX_ROWS", "0"))
    if max_rows > 0:
        merged = merged.head(max_rows)
    print(f"[KAFKA PRODUCER] Publishing rows={len(merged)} to topic={topic}")

    producer = AIOKafkaProducer(
        bootstrap_servers=kafka_bootstrap,
        acks="all",
        linger_ms=20,
        # aiokafka uses `max_batch_size` (not `batch_size`)
        max_batch_size=32768,
    )
    await producer.start()
    try:
        for i, row in enumerate(merged.to_dict(orient="records")):
            payload: Dict[str, Any] = row
            message = json.dumps(payload, default=str).encode("utf-8")
            await producer.send_and_wait(topic, message)
            if (i + 1) % 10000 == 0:
                print(f"[KAFKA PRODUCER] sent {i+1}")
    finally:
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(produce_transactions())

