"""
ATLAS-X Alert Consumer – Fraud Alert Logger
────────────────────────────────────────────
Subscribes to 'fraud-alerts' and 'fraud-blocks', logs each alert to
Postgres, and prints a human-readable line per alert.

Output format:
  ⚠️  FLAGGED: txn_12345 | prob=0.78 | graph_risk=0.65 | seg=VIP
  🚫 BLOCKED: txn_67890 | prob=0.95 | graph_risk=0.82 | seg=Regular

Usage:
    python -m src.streaming.alert_consumer

Environment:
    KAFKA_BOOTSTRAP_SERVERS  default: localhost:29092
    KAFKA_TOPIC_FRAUD_ALERTS default: fraud-alerts
    KAFKA_TOPIC_FRAUD_BLOCKS default: fraud-blocks
    KAFKA_ALERT_GROUP        default: alert-logger-v1
    POSTGRES_DSN             default: (empty = no DB logging)
"""
import asyncio
import json
import os
from typing import Optional

import asyncpg
from aiokafka import AIOKafkaConsumer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS",  "localhost:29092")
TOPIC_ALERTS    = os.getenv("KAFKA_TOPIC_FRAUD_ALERTS",  "fraud-alerts")
TOPIC_BLOCKS    = os.getenv("KAFKA_TOPIC_FRAUD_BLOCKS",  "fraud-blocks")
CONSUMER_GROUP  = os.getenv("KAFKA_ALERT_GROUP",         "alert-logger-v1")
POSTGRES_DSN    = os.getenv("POSTGRES_DSN",              "")


async def _log_alert(pg: Optional[asyncpg.Connection], data: dict) -> None:
    if pg is None:
        return
    try:
        await pg.execute(
            """
            INSERT INTO fraud_alerts
                (transaction_id, fraud_probability, graph_risk_score,
                 decision, customer_segment, is_fraud_label)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT DO NOTHING
            """,
            str(data.get("transaction_id", "")),
            float(data.get("fraud_probability", 0.0)),
            float(data.get("graph_risk_score",  0.0)),
            str(data.get("decision", "")),
            str(data.get("customer_segment", "")),
            data.get("is_fraud_label"),
        )
    except Exception as e:
        print(f"[alert-db] write error: {e}", flush=True)


def _format_alert(data: dict) -> str:
    txn_id    = data.get("transaction_id", "?")
    prob      = data.get("fraud_probability", 0.0)
    risk      = data.get("graph_risk_score",  0.0)
    seg       = data.get("customer_segment",  "?")
    decision  = data.get("decision", "")
    label     = data.get("is_fraud_label")
    label_str = f" | GT={'fraud' if label else 'legit'}" if label is not None else ""

    if decision == "FLAG":
        icon = "⚠️ "
        tag  = "FLAGGED"
    else:
        icon = "🚫"
        tag  = "BLOCKED"

    return (
        f"{icon} {tag}: txn={txn_id} | "
        f"prob={prob:.3f} | "
        f"graph_risk={risk:.3f} | "
        f"seg={seg}"
        f"{label_str}"
    )


async def consume_alerts() -> None:
    # ── Postgres ──────────────────────────────────────────────────────────────
    pg: Optional[asyncpg.Connection] = None
    if POSTGRES_DSN:
        try:
            pg = await asyncpg.connect(POSTGRES_DSN)
            print("[alert] Postgres connected")
        except Exception as e:
            print(f"[alert] Postgres unavailable: {e}")

    # ── Kafka ──────────────────────────────────────────────────────────────────
    consumer = AIOKafkaConsumer(
        TOPIC_ALERTS,
        TOPIC_BLOCKS,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=CONSUMER_GROUP,
        enable_auto_commit=True,
        auto_offset_reset="earliest",
    )
    await consumer.start()
    print(f"[alert] Subscribed to '{TOPIC_ALERTS}' + '{TOPIC_BLOCKS}'")
    print(f"[alert] Waiting for alerts...\n")

    flag_count  = 0
    block_count = 0

    try:
        async for msg in consumer:
            try:
                data = json.loads(msg.value.decode("utf-8"))
            except Exception:
                continue

            topic = msg.topic
            if topic == TOPIC_ALERTS:
                flag_count += 1
                data["decision"] = "FLAG"
            else:
                block_count += 1
                data["decision"] = "BLOCK"

            print(_format_alert(data), flush=True)
            await _log_alert(pg, data)

    except asyncio.CancelledError:
        pass
    finally:
        await consumer.stop()
        if pg:
            await pg.close()
        print(
            f"\n[alert] Stopped.  "
            f"Flags={flag_count:,}  Blocks={block_count:,}  "
            f"Total={flag_count + block_count:,}"
        )


if __name__ == "__main__":
    asyncio.run(consume_alerts())
