"""
ATLAS-X Kafka Consumer – Fraud Detector
─────────────────────────────────────────
Subscribes to 'transactions', calls the FastAPI /api/v1/predict endpoint
for each message, then routes by decision:

  APPROVE → log to Postgres only
  FLAG    → publish to 'fraud-alerts'  + log to Postgres
  BLOCK   → publish to 'fraud-blocks'  + log to Postgres

Progress printed every 100 transactions:
  "Processed 500 txns | 87 txns/sec | Avg latency: 45ms | Lag: 23"

Usage:
    python -m src.streaming.kafka_consumer

Environment:
    KAFKA_BOOTSTRAP_SERVERS  default: localhost:29092
    KAFKA_TOPIC_TRANSACTIONS default: transactions
    KAFKA_TOPIC_FRAUD_ALERTS default: fraud-alerts
    KAFKA_TOPIC_FRAUD_BLOCKS default: fraud-blocks
    KAFKA_CONSUMER_GROUP     default: fraud-detector-v2
    ATLAS_API_URL            default: http://localhost:8000
    POSTGRES_DSN             default: (empty = no DB logging)
"""
import asyncio
import json
import os
import time
from collections import deque
from typing import Optional

import aiohttp
import asyncpg
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS",  "localhost:29092")
TOPIC_IN        = os.getenv("KAFKA_TOPIC_TRANSACTIONS",  "transactions")
TOPIC_ALERTS    = os.getenv("KAFKA_TOPIC_FRAUD_ALERTS",  "fraud-alerts")
TOPIC_BLOCKS    = os.getenv("KAFKA_TOPIC_FRAUD_BLOCKS",  "fraud-blocks")
CONSUMER_GROUP  = os.getenv("KAFKA_CONSUMER_GROUP",      "fraud-detector-v2")
API_URL         = os.getenv("ATLAS_API_URL",             "http://localhost:8001")
POSTGRES_DSN    = os.getenv("POSTGRES_DSN",              "")
LOG_INTERVAL    = 100     # print stats every N transactions
IN_FLIGHT_MAX   = 50      # max concurrent API calls


async def _log_to_postgres(
    pg: Optional[asyncpg.Connection],
    result: dict,
) -> None:
    if pg is None:
        return
    try:
        await pg.execute(
            """
            INSERT INTO predictions
                (transaction_id, fraud_probability, graph_risk_score,
                 decision, latency_ms)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT DO NOTHING
            """,
            result["transaction_id"],
            result["fraud_probability"],
            result["graph_risk_score"],
            result["decision"],
            result["latency_ms"],
        )
    except Exception as e:
        print(f"[DB] write error: {e}", flush=True)


async def _call_predict(
    session: aiohttp.ClientSession,
    message: dict,
) -> Optional[dict]:
    """POST to /api/v1/predict and return the response dict."""
    payload = {
        "transaction_id": message.get("transaction_id", "unknown"),
        "card_id":        message.get("card_id", "?-?-?"),
        "amount":         float(message.get("amount", 0.0)),
        "features":       message.get("features", {}),
    }
    try:
        async with session.post(
            f"{API_URL}/api/v1/predict",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"[API] {resp.status}: {text[:120]}", flush=True)
                return None
            return await resp.json()
    except Exception as e:
        print(f"[API] call failed: {e}", flush=True)
        return None


async def consume() -> None:
    # ── Verify API is reachable ───────────────────────────────────────────────
    async with aiohttp.ClientSession() as sess:
        try:
            async with sess.get(f"{API_URL}/healthz", timeout=aiohttp.ClientTimeout(total=5)) as r:
                assert r.status == 200
            print(f"[consumer] API reachable at {API_URL}")
        except Exception as e:
            print(f"[consumer] ERROR: cannot reach API at {API_URL}: {e}")
            print("  Start the API first:  ./run_api.sh")
            return

    # ── Postgres ──────────────────────────────────────────────────────────────
    pg: Optional[asyncpg.Connection] = None
    if POSTGRES_DSN:
        try:
            pg = await asyncpg.connect(POSTGRES_DSN)
            print(f"[consumer] Postgres connected")
        except Exception as e:
            print(f"[consumer] Postgres unavailable: {e}")

    # ── Kafka setup ───────────────────────────────────────────────────────────
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        acks="all",
        linger_ms=5,
    )
    consumer = AIOKafkaConsumer(
        TOPIC_IN,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=CONSUMER_GROUP,
        enable_auto_commit=True,
        auto_offset_reset="earliest",
    )
    await producer.start()
    await consumer.start()
    print(f"[consumer] Subscribed to '{TOPIC_IN}' | group={CONSUMER_GROUP}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    processed   = 0
    decisions   = {"APPROVE": 0, "FLAG": 0, "BLOCK": 0}
    latencies: deque[float] = deque(maxlen=500)
    t_start     = time.perf_counter()
    semaphore   = asyncio.Semaphore(IN_FLIGHT_MAX)

    async def handle(raw_msg: bytes) -> None:
        nonlocal processed
        async with semaphore:
            t0 = time.perf_counter()
            try:
                message = json.loads(raw_msg.decode("utf-8"))
            except Exception:
                return

            async with aiohttp.ClientSession() as sess:
                result = await _call_predict(sess, message)

            if result is None:
                return

            e2e_ms = (time.perf_counter() - t0) * 1000
            latencies.append(e2e_ms)

            decision = result.get("decision", "APPROVE")
            if decision in decisions:
                decisions[decision] += 1
            processed += 1

            # ── Route downstream ──────────────────────────────────────────────
            alert_payload = json.dumps({
                **result,
                "is_fraud_label": message.get("is_fraud"),
                "e2e_latency_ms": round(e2e_ms, 2),
            }, default=str).encode("utf-8")

            if decision == "FLAG":
                await producer.send_and_wait(TOPIC_ALERTS, alert_payload)
            elif decision == "BLOCK":
                await producer.send_and_wait(TOPIC_BLOCKS, alert_payload)

            await _log_to_postgres(pg, result)

            # Progress log
            if processed % LOG_INTERVAL == 0:
                elapsed = max(1e-6, time.perf_counter() - t_start)
                tps     = processed / elapsed
                avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
                # Consumer lag via partition assignment (approximate)
                try:
                    tps_parts = consumer.assignment()
                    lag_total = sum(
                        (await consumer.end_offsets([tp]))[tp]
                        - consumer.last_stable_offset(tp)
                        for tp in tps_parts
                    )
                except Exception:
                    lag_total = -1

                lag_str = f"{lag_total}" if lag_total >= 0 else "n/a"
                print(
                    f"Processed {processed:,} txns | "
                    f"{tps:.0f} txns/sec | "
                    f"Avg latency: {avg_lat:.0f}ms | "
                    f"Lag: {lag_str}",
                    flush=True,
                )

    # ── Main consume loop ─────────────────────────────────────────────────────
    tasks: set[asyncio.Task] = set()
    try:
        async for msg in consumer:
            task = asyncio.create_task(handle(msg.value))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

            # Bound in-flight tasks
            if len(tasks) > IN_FLIGHT_MAX * 2:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    except asyncio.CancelledError:
        pass
    finally:
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await consumer.stop()
        await producer.stop()
        if pg:
            await pg.close()

        elapsed = max(1e-6, time.perf_counter() - t_start)
        print(f"\n[consumer] Stopped after {processed:,} txns in {elapsed:.1f}s")
        print(f"  Decisions: {decisions}")


if __name__ == "__main__":
    asyncio.run(consume())
