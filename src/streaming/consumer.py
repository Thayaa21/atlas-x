import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import asyncpg
from aiokafka import AIOKafkaConsumer, ConsumerRecord


@dataclass(frozen=True)
class ConsumerConfig:
    kafka_bootstrap: str
    topic_in: str
    topic_out: str
    consumer_group: str
    postgres_dsn: Optional[str]


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "y", "on")


def _get_env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v


def load_consumer_config() -> ConsumerConfig:
    kafka_bootstrap = _get_env("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
    topic_in = _get_env("KAFKA_TOPIC_TRANSACTIONS", "transactions")
    topic_out = _get_env("KAFKA_TOPIC_FRAUD_ALERTS", "fraud-alerts")
    consumer_group = _get_env("KAFKA_CONSUMER_GROUP", "fraud-consumer-v1")
    postgres_dsn = _get_env("POSTGRES_DSN", "")
    return ConsumerConfig(
        kafka_bootstrap=kafka_bootstrap,
        topic_in=topic_in,
        topic_out=topic_out,
        consumer_group=consumer_group,
        postgres_dsn=postgres_dsn or None,
    )


async def maybe_write_audit(pg_conn, row: Dict[str, Any]) -> None:
    if pg_conn is None:
        return
    # Keep schema simple for now.
    await pg_conn.execute(
        """
        INSERT INTO transaction_audit (transaction_id, fraud_prob, decision, segment, dqn_action, created_at)
        VALUES ($1, $2, $3, $4, $5, NOW())
        ON CONFLICT (transaction_id) DO NOTHING
        """,
        int(row.get("TransactionID", 0)),
        float(row.get("fraud_prob", 0.0)),
        row.get("decision", ""),
        row.get("segment", ""),
        row.get("dqn_action", ""),
    )


async def consume_and_score() -> None:
    cfg = load_consumer_config()

    demo_lookup = env_flag("STREAMING_DEMO_LOOKUP", "0")

    server = None
    demo = None
    if not demo_lookup:
        # Import here so startup is not blocked by heavy imports in environments without FastAPI.
        from src.api.main import FraudAPIServer

        server = FraudAPIServer()
    else:
        # Fast path for local end-to-end testing:
        # Use precomputed `train_full_features.parquet` and skip the heavy online FeaturePipeline.
        from pathlib import Path
        import joblib
        import torch
        import numpy as np
        import pandas as pd

        repo_root = Path(__file__).resolve().parents[2]
        model_p = repo_root / "src/models/atlass_x_xgb_v3.pkl"
        thresholds_p = repo_root / "src/optimization/artifacts/thresholds.json"
        dqn_p = repo_root / "src/rl/trained_dqn.pth"
        full_features_p = repo_root / "data/processed/train_full_features.parquet"

        print("[DEMO] Loading model artifacts...", flush=True)
        if not model_p.exists():
            raise FileNotFoundError(f"Missing model: {model_p}")
        if not thresholds_p.exists():
            raise FileNotFoundError(f"Missing thresholds artifact: {thresholds_p}")
        if not dqn_p.exists():
            raise FileNotFoundError(f"Missing DQN: {dqn_p}")
        if not full_features_p.exists():
            raise FileNotFoundError(f"Missing full features: {full_features_p}")

        model = joblib.load(model_p)
        thresholds_art = json.loads(thresholds_p.read_text())
        seg_params = thresholds_art["segment_quantile_cuts"]
        thresholds = {seg: float(thresholds_art["segments"][seg]["threshold"]) for seg in ["VIP", "Regular", "New"]}

        from src.rl.dqn_agent import ACTIONS, QNetwork, encode_segment

        ckpt = torch.load(dqn_p, map_location="cpu")
        qnet = QNetwork(int(ckpt["input_dim"]), int(ckpt["output_dim"]))
        qnet.load_state_dict(ckpt["state_dict"])
        qnet.eval()

        # Load full features once; index by TransactionID for fast lookup.
        print("[DEMO] Loading train_full_features.parquet (this can take a bit)...", flush=True)
        full_df = pd.read_parquet(full_features_p)
        if "TransactionID" not in full_df.columns:
            raise ValueError("Expected TransactionID column in train_full_features.parquet")
        full_df = full_df.set_index("TransactionID", drop=True)
        print(f"[DEMO] Loaded full features rows={len(full_df)}", flush=True)

        feature_names = list(model.feature_names_in_)
        # Match training-time categorical casting:
        # cast all categorical/object columns once on the full dataframe so the category codes match.
        cols_to_cast = [c for c in feature_names if c in full_df.columns and full_df[c].dtype == object]
        for col in cols_to_cast:
            full_df[col] = full_df[col].astype("object").fillna("None").astype("category")

        def segment_from_cuts_local(d1: float, d15: float) -> str:
            eps = 1e-12
            d1_min = float(seg_params["d1_min"])
            d1_max = float(seg_params["d1_max"])
            d15_min = float(seg_params["d15_min"])
            d15_max = float(seg_params["d15_max"])
            vip_cut = float(seg_params["vip_cut"])
            regular_cut = float(seg_params["regular_cut"])

            tenure_score = (d1 - d1_min) / (d1_max - d1_min + eps)
            recency_score = (d15_max - d15) / (d15_max - d15_min + eps)
            combined = 0.5 * tenure_score + 0.5 * recency_score
            if combined >= vip_cut:
                return "VIP"
            if combined >= regular_cut:
                return "Regular"
            return "New"

        def score_one(tx_payload: Dict[str, Any]) -> Dict[str, Any]:
            txid = int(tx_payload.get("TransactionID", 0))
            if txid not in full_df.index:
                raise KeyError(f"TransactionID {txid} not found in full feature dataset")
            row = full_df.loc[txid]

            # Use DataFrame slice to preserve categorical dtype and category codes.
            X_df = full_df.loc[[txid], feature_names]
            fraud_prob = float(model.predict_proba(X_df)[:, 1][0])
            d1 = float(row["D1"])
            d15 = float(row["D15"])
            segment = segment_from_cuts_local(d1=d1, d15=d15)
            threshold = float(thresholds[segment])
            decision = "FRAUD" if fraud_prob >= threshold else "CLEAR"

            graph_risk = float(row["cluster_fraud_rate"])
            market_context = float(row["Transaction_Hour"] / 23.0)
            seg_code = float(encode_segment(segment))
            state_vec = np.array([fraud_prob, graph_risk, seg_code, market_context], dtype=np.float32)

            with torch.no_grad():
                q = qnet(torch.from_numpy(state_vec).unsqueeze(0))
                action_idx = int(torch.argmax(q, dim=1).item())
            action = ACTIONS[action_idx]

            return {
                "TransactionID": txid,
                "fraud_prob": fraud_prob,
                "segment": segment,
                "threshold": threshold,
                "decision": decision,
                "graph_risk": graph_risk,
                "market_context": market_context,
                "dqn_action": action,
            }

        demo = {"score_one": score_one}

    producer = None
    consumer = None
    pg_conn = None
    try:
        if cfg.postgres_dsn:
            pg_conn = await asyncpg.connect(cfg.postgres_dsn)

        from aiokafka import AIOKafkaProducer

        producer = AIOKafkaProducer(
            bootstrap_servers=cfg.kafka_bootstrap,
            acks="all",
            linger_ms=20,
            max_batch_size=32768,
        )
        await producer.start()

        consumer = AIOKafkaConsumer(
            cfg.topic_in,
            bootstrap_servers=cfg.kafka_bootstrap,
            group_id=cfg.consumer_group,
            enable_auto_commit=True,
            auto_offset_reset=os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest"),
        )
        await consumer.start()

        in_flight = 0
        start = time.time()
        processed = 0

        async def handle_record(rec: ConsumerRecord):
            nonlocal processed, in_flight
            in_flight += 1
            t0 = time.time()
            try:
                payload = json.loads(rec.value.decode("utf-8"))
                # Compute full decision + DQN action.
                try:
                    if demo_lookup:
                        scored = demo["score_one"](payload.get("transaction", payload))
                    else:
                        scored = server.dqn_action(payload.get("transaction", payload))
                except Exception as e:
                    txid = payload.get("TransactionID", None)
                    print(f"[CONSUMER] scoring failed txid={txid}: {type(e).__name__}: {e}", flush=True)
                    return

                out_msg = {
                    "TransactionID": scored.get("TransactionID"),
                    "fraud_prob": scored.get("fraud_prob"),
                    "segment": scored.get("segment"),
                    "threshold": scored.get("threshold"),
                    "decision": scored.get("decision"),
                    "dqn_action": scored.get("dqn_action"),
                    "graph_risk": scored.get("graph_risk"),
                    "market_context": scored.get("market_context"),
                    "timestamp": time.time(),
                }
                msg_bytes = json.dumps(out_msg, default=str).encode("utf-8")
                await producer.send_and_wait(cfg.topic_out, msg_bytes)
                await maybe_write_audit(pg_conn, scored)
                processed += 1

                if processed <= int(os.getenv("DEMO_LOG_FIRST_N", "5")):
                    print(
                        f"[CONSUMER] published txid={out_msg['TransactionID']} decision={out_msg['decision']} dqn={out_msg['dqn_action']}",
                        flush=True,
                    )
            finally:
                in_flight -= 1
                _ = time.time() - t0

        loop = asyncio.get_event_loop()
        tasks = set()

        async for msg in consumer:
            task = asyncio.create_task(handle_record(msg))
            tasks.add(task)
            # Backpressure: keep a bounded amount of in-flight work.
            if len(tasks) > int(os.getenv("KAFKA_IN_FLIGHT_LIMIT", "200")):
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = pending

            if processed > 0 and processed % 5000 == 0:
                elapsed = max(1e-6, time.time() - start)
                tps = processed / elapsed
                print(f"[KAFKA CONSUMER] processed={processed} tps={tps:.1f} in_flight={in_flight}")
    finally:
        if consumer is not None:
            await consumer.stop()
        if producer is not None:
            await producer.stop()
        if pg_conn is not None:
            await pg_conn.close()


if __name__ == "__main__":
    asyncio.run(consume_and_score())

