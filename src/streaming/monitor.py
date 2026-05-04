"""
ATLAS-X Streaming Monitor
──────────────────────────
Real-time CLI dashboard. Polls the API /api/v1/stats endpoint and
optionally queries Postgres for alert counts.

Updates every 5 seconds (configurable via --interval).

Usage:
    python -m src.streaming.monitor [--interval 5] [--api http://localhost:8000]

╔═══════════════════════════════════════════════════╗
║  ATLAS-X STREAMING MONITOR                        ║
╠═══════════════════════════════════════════════════╣
║  Throughput:  127 txns/sec                        ║
║  Latency:     avg=18ms                            ║
║  Decisions:   APPROVE: 89  FLAG: 8  BLOCK: 3     ║
║  Fraud Rate:  3.2%                                ║
╚═══════════════════════════════════════════════════╝
"""
import argparse
import asyncio
import os
import sys
import time
from typing import Optional

import aiohttp

ATLAS_API_URL = os.getenv("ATLAS_API_URL", "http://localhost:8000")
POSTGRES_DSN  = os.getenv("POSTGRES_DSN", "")


def _clear():
    print("\033[2J\033[H", end="", flush=True)


def _render(stats: dict, alert_counts: Optional[dict],
            elapsed_since_start: float, prev_total: int) -> int:
    total   = stats.get("total_predictions", 0)
    fraud_r = stats.get("fraud_rate", 0.0)
    avg_lat = stats.get("avg_latency_ms", 0.0)
    decs    = stats.get("decisions", {})

    approves = decs.get("APPROVE", 0)
    flags    = decs.get("FLAG",    0)
    blocks   = decs.get("BLOCK",   0)

    # Approximate current throughput from delta since last render
    delta_txns = total - prev_total

    _clear()
    W = 51
    print("╔" + "═" * W + "╗")
    print("║  ATLAS-X STREAMING MONITOR" + " " * (W - 26) + "║")
    print("╠" + "═" * W + "╣")
    print(f"║  {'Total predictions:':<20} {total:>10,}" + " " * (W - 34) + "║")
    print(f"║  {'Delta (this window):':<20} {delta_txns:>+10,}" + " " * (W - 34) + "║")
    print(f"║  {'Avg latency:':<20} {avg_lat:>9.1f}ms" + " " * (W - 33) + "║")
    print(f"║  {'Fraud rate:':<20} {fraud_r:>9.1%}" + " " * (W - 32) + "║")
    print("╠" + "═" * W + "╣")
    print(f"║  {'Decisions'}" + " " * (W - 10) + "║")
    print(f"║    APPROVE: {approves:>7,}  FLAG: {flags:>5,}  BLOCK: {blocks:>5,}" + " " * (W - 44) + "║")
    if alert_counts:
        db_flags  = alert_counts.get("flags",  0)
        db_blocks = alert_counts.get("blocks", 0)
        print(f"║  DB alerts: FLAG={db_flags:,}  BLOCK={db_blocks:,}" + " " * max(0, W - 35) + "║")
    print("╠" + "═" * W + "╣")
    ts = time.strftime("%H:%M:%S")
    print(f"║  Updated: {ts}" + " " * (W - 20) + "║")
    print("╚" + "═" * W + "╝")
    print("  (Ctrl-C to stop)")
    return total


async def _fetch_stats(session: aiohttp.ClientSession, api_url: str) -> Optional[dict]:
    try:
        async with session.get(
            f"{api_url}/api/v1/stats",
            timeout=aiohttp.ClientTimeout(total=3),
        ) as r:
            if r.status == 200:
                return await r.json()
    except Exception:
        pass
    return None


async def _fetch_alert_counts(dsn: str) -> Optional[dict]:
    if not dsn:
        return None
    try:
        import asyncpg
        conn = await asyncpg.connect(dsn)
        try:
            flags  = await conn.fetchval(
                "SELECT COUNT(*) FROM fraud_alerts WHERE decision='FLAG'"
            )
            blocks = await conn.fetchval(
                "SELECT COUNT(*) FROM fraud_alerts WHERE decision='BLOCK'"
            )
            return {"flags": int(flags or 0), "blocks": int(blocks or 0)}
        finally:
            await conn.close()
    except Exception:
        return None


async def monitor(api_url: str, interval: float) -> None:
    print(f"ATLAS-X Monitor  |  API={api_url}  |  refresh={interval}s")
    print("Connecting...")

    t_start   = time.perf_counter()
    prev_total = 0

    async with aiohttp.ClientSession() as session:
        while True:
            stats = await _fetch_stats(session, api_url)
            if stats is None:
                _clear()
                print(f"[monitor] Cannot reach {api_url}/api/v1/stats — retrying in {interval}s")
            else:
                alert_counts = await _fetch_alert_counts(POSTGRES_DSN)
                elapsed      = time.perf_counter() - t_start
                prev_total   = _render(stats, alert_counts, elapsed, prev_total)

            await asyncio.sleep(interval)


def _parse():
    p = argparse.ArgumentParser(description="ATLAS-X streaming monitor")
    p.add_argument("--api",      default=ATLAS_API_URL)
    p.add_argument("--interval", type=float, default=5.0,
                   help="Refresh interval in seconds (default 5)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    try:
        asyncio.run(monitor(args.api, args.interval))
    except KeyboardInterrupt:
        print("\n[monitor] Stopped.")
