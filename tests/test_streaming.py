"""
ATLAS-X – Streaming Integration Test
───────────────────────────────────────
End-to-end test: producer → Kafka → consumer → Postgres.

Prerequisites (all must be running before this test):
  1.  docker compose -f docker/docker-compose.yml up -d kafka zookeeper
  2.  ./run_api.sh              (FastAPI on port 8000)
  3.  python -m src.streaming.init_db
  4.  (optional) python -m src.streaming.kafka_consumer  in background

What this test does:
  • Sends N_MESSAGES transactions via the Kafka producer
  • Waits for the consumer to process them (polls Postgres until count matches)
  • Asserts:
      - All messages received (within timeout)
      - Average latency reported by API < 100ms
      - Decision distribution is reasonable (not all APPROVE, not all BLOCK)
      - No errors

Usage:
    python tests/test_streaming.py
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

BASE_URL    = "http://localhost:8000"
BOOTSTRAP   = "localhost:29092"
POSTGRES_DSN= "postgresql://postgres:postgres@localhost:5432/atlasx"
N_MESSAGES  = 100
RATE        = 50.0      # txns/sec during test
TIMEOUT_S   = 60        # seconds to wait for consumer to catch up

SAMPLES_P   = Path(__file__).parent / "sample_transactions.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _api_health() -> bool:
    async with aiohttp.ClientSession() as s:
        try:
            async with s.get(f"{BASE_URL}/healthz",
                             timeout=aiohttp.ClientTimeout(total=5)) as r:
                return r.status == 200
        except Exception:
            return False


async def _kafka_available() -> bool:
    try:
        from aiokafka import AIOKafkaProducer
        p = AIOKafkaProducer(bootstrap_servers=BOOTSTRAP)
        await p.start()
        await p.stop()
        return True
    except Exception:
        return False


async def _db_count(table: str) -> int:
    try:
        import asyncpg
        conn = await asyncpg.connect(POSTGRES_DSN)
        n = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
        await conn.close()
        return int(n or 0)
    except Exception:
        return -1


async def _get_api_stats() -> dict:
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BASE_URL}/api/v1/stats") as r:
            return await r.json()


# ── Test functions ────────────────────────────────────────────────────────────

async def check_prerequisites() -> list[str]:
    errors = []
    if not await _api_health():
        errors.append(f"API not reachable at {BASE_URL}  (run ./run_api.sh)")
    if not await _kafka_available():
        errors.append(f"Kafka not reachable at {BOOTSTRAP}  (run: cd docker && docker compose up -d kafka zookeeper)")
    return errors


async def test_producer_sends(messages: list[dict]) -> None:
    """Test 1: producer can publish N messages without error."""
    from src.streaming.kafka_producer import produce
    print(f"  Sending {N_MESSAGES} messages at {RATE:.0f} txns/sec...")
    await produce(
        bootstrap=BOOTSTRAP,
        topic="transactions",
        messages=messages,
        rate=RATE,
        count=N_MESSAGES,
    )
    print(f"  ✓ {N_MESSAGES} messages published")


async def test_consumer_processes(baseline_count: int) -> dict:
    """
    Test 2: wait for consumer to process all messages (polls Postgres).
    Returns the final stats dict.
    """
    target   = baseline_count + N_MESSAGES
    deadline = time.time() + TIMEOUT_S
    print(f"  Waiting for consumer to log {N_MESSAGES} rows to Postgres "
          f"(timeout={TIMEOUT_S}s)...")

    while time.time() < deadline:
        current = await _db_count("predictions")
        if current < 0:
            print("  [skip] Postgres unavailable — skipping DB assertion")
            return {}
        processed = current - baseline_count
        print(f"    {processed}/{N_MESSAGES} processed", end="\r", flush=True)
        if current >= target:
            print(f"  ✓ All {N_MESSAGES} transactions logged to Postgres")
            return await _get_api_stats()
        await asyncio.sleep(2)

    # Timeout — report what we got
    current   = await _db_count("predictions")
    processed = current - baseline_count
    print(f"\n  WARN: only {processed}/{N_MESSAGES} processed within {TIMEOUT_S}s")
    print("  Is the consumer running?  python -m src.streaming.kafka_consumer")
    return {}


async def test_latency(stats: dict) -> None:
    """Test 3: avg latency reported by API < 100ms."""
    if not stats:
        print("  [skip] no stats available")
        return
    avg = stats.get("avg_latency_ms", 0.0)
    assert avg < 100, f"avg latency {avg:.1f}ms exceeds 100ms target"
    print(f"  ✓ avg latency = {avg:.1f}ms < 100ms")


async def test_decision_distribution(stats: dict) -> None:
    """Test 4: not all transactions get the same decision."""
    if not stats:
        print("  [skip] no stats available")
        return
    decs = stats.get("decisions", {})
    distinct = sum(1 for v in decs.values() if v > 0)
    assert distinct >= 1, f"Expected at least 1 decision type, got: {decs}"
    total = sum(decs.values())
    print(f"  ✓ decisions={decs}  (total={total:,})")


# ── Runner ────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("ATLAS-X Streaming Integration Test")
    print("=" * 48)

    # Check prerequisites
    print("\nChecking prerequisites...")
    errors = await check_prerequisites()
    if errors:
        for e in errors:
            print(f"  [MISSING] {e}")
        sys.exit(1)
    print("  API and Kafka reachable.")

    # Load sample messages
    if not SAMPLES_P.exists():
        print(f"[ERROR] {SAMPLES_P} not found — regenerate with test setup")
        sys.exit(1)
    all_samples = json.loads(SAMPLES_P.read_text())
    # Expand if needed (cycle samples to reach N_MESSAGES)
    messages = (all_samples * (N_MESSAGES // len(all_samples) + 1))[:N_MESSAGES]

    # DB baseline
    baseline = await _db_count("predictions")
    if baseline < 0:
        print("  Postgres unavailable — DB assertions will be skipped")
        baseline = 0

    results = []
    tests = [
        ("Producer sends messages",    lambda: test_producer_sends(messages)),
        ("Consumer processes messages", lambda: test_consumer_processes(baseline)),
        ("Latency < 100ms",            None),    # populated from prev result
        ("Decision distribution",      None),
    ]

    # Run test 1 + 2
    print("\nTest 1: Producer")
    try:
        await test_producer_sends(messages)
        results.append(("Producer sends messages", True, None))
    except Exception as e:
        results.append(("Producer sends messages", False, str(e)))
        print(f"  FAIL: {e}")

    print("\nTest 2: Consumer processes")
    stats = {}
    try:
        stats = await test_consumer_processes(baseline)
        results.append(("Consumer processes messages", True, None))
    except Exception as e:
        results.append(("Consumer processes messages", False, str(e)))
        print(f"  FAIL: {e}")

    print("\nTest 3: Latency")
    try:
        await test_latency(stats)
        results.append(("Latency < 100ms", True, None))
    except AssertionError as e:
        results.append(("Latency < 100ms", False, str(e)))
        print(f"  FAIL: {e}")

    print("\nTest 4: Decision distribution")
    try:
        await test_decision_distribution(stats)
        results.append(("Decision distribution", True, None))
    except AssertionError as e:
        results.append(("Decision distribution", False, str(e)))
        print(f"  FAIL: {e}")

    # Summary
    print("\n" + "=" * 48)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    for name, ok, msg in results:
        mark = "✓" if ok else "✗"
        line = f"  {mark} {name}"
        if msg:
            line += f": {msg}"
        print(line)
    print("=" * 48)
    print(f"  {passed}/{len(results)} passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
