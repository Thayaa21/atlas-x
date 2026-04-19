"""
ATLAS-X – DQN v2 Training
──────────────────────────
Trains the graph-augmented DQN agent on the 20% holdout.

Data flow:
  1. Load train_with_graph_features.parquet (has graph_risk_score column)
  2. Reproduce 20% holdout split (same random_state=42 as model training)
  3. Run v4 model to get fraud_prob for each test transaction
  4. Build FraudEnv and run 50,000 episodes

Saves:
  src/rl/trained_dqn_v2.pth
  results/dqn_training_metrics.json

Usage:
    python -m src.rl.train_dqn_v2
"""
import json
import os
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.optimization.threshold_optimizer import compute_customer_segments
from src.rl.dqn_agent_v2 import ACTIONS, DQNAgentV2
from src.rl.fraud_env import FraudEnv

REPO      = Path(__file__).resolve().parents[2]
DATA_P    = REPO / "data/processed/train_with_graph_features.parquet"
MODEL_P   = REPO / "src/models/atlass_x_xgb_v4_graph.pkl"
OUT_MODEL = REPO / "src/rl/trained_dqn_v2.pth"
OUT_JSON  = REPO / "results/dqn_training_metrics.json"
OUT_PLOT  = REPO / "results/dqn_reward_curve.png"

EPISODES     = int(os.getenv("DQN_EPISODES", "50000"))
BATCH_SIZE   = 64
LR           = 1e-3
BUFFER_SIZE  = 10_000
LOG_EVERY    = 5_000


def build_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (test_df, fraud_prob) for the 20% holdout."""
    from sklearn.model_selection import StratifiedShuffleSplit

    segment_s, _ = compute_customer_segments(df)
    df = df.copy()
    df["customer_segment"] = segment_s

    drop_cols = ["isFraud", "TransactionID", "TransactionDT", "customer_segment"]
    X = df.drop([c for c in drop_cols if c in df.columns], axis=1)
    y = df["isFraud"].astype(int).to_numpy()

    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].astype("object").fillna("None").astype("category")

    graph_cols = ["device_fraud_rate", "device_card_velocity", "connected_fraud_cards",
                  "email_fraud_rate", "address_fraud_rate", "graph_risk_score"]
    for col in graph_cols:
        if col in X.columns:
            X[col] = X[col].fillna(0.0)

    # Use index-based split to avoid copying the full dataframe
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(sss.split(X, y))

    X_test  = X.iloc[test_idx].reset_index(drop=True)
    y_test  = y[test_idx]
    df_test = df.iloc[test_idx].reset_index(drop=True)

    print("Running v4 model on holdout...")
    model      = joblib.load(MODEL_P)
    fraud_prob = model.predict_proba(X_test)[:, 1].astype(np.float32)

    return df_test, fraud_prob


def plot_reward_curve(ep_rewards: list, window: int, path: Path) -> None:
    arr = np.array(ep_rewards, dtype=np.float32)
    smoothed = np.convolve(arr, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(arr, alpha=0.25, color="#4C72B0", label="Episode reward")
    ax.plot(range(window - 1, len(arr)), smoothed, color="#4C72B0",
            linewidth=1.8, label=f"{window}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward ($)")
    ax.set_title("DQN v2 Training – Reward Convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not DATA_P.exists():
        raise FileNotFoundError(
            f"{DATA_P} not found.\n"
            "Run  python -m src.features.engineer_graph_features  first."
        )
    if not MODEL_P.exists():
        raise FileNotFoundError(
            f"{MODEL_P} not found.\n"
            "Run  python -m src.models.train_v4_with_graph  first."
        )

    np.random.seed(42)

    print(f"Loading {DATA_P.name}...")
    df = pd.read_parquet(DATA_P)
    print(f"  {len(df):,} transactions, {int(df['isFraud'].sum()):,} fraud")

    df_test, fraud_prob = build_holdout(df)
    print(f"  Holdout: {len(df_test):,} rows, fraud rate={df_test['isFraud'].mean():.3%}")

    env = FraudEnv.from_dataframe(df_test, fraud_prob, seed=42)

    agent = DQNAgentV2(
        lr=LR,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_steps=int(EPISODES * 0.8),
        target_update_every=1_000,
    )

    ep_rewards: list[float] = []
    action_counts = np.zeros(4, dtype=np.int64)
    losses: list[float] = []

    print(f"\nTraining for {EPISODES:,} episodes  (batch={BATCH_SIZE}, buffer={BUFFER_SIZE:,})...")
    t0 = time.perf_counter()

    for ep in range(EPISODES):
        state  = env.reset()
        action = agent.select_action(state)
        next_state, reward, _, _ = env.step(action)

        agent.push(state, action, reward, next_state, done=0.0)
        loss = agent.train_step()

        ep_rewards.append(reward)
        action_counts[action] += 1
        if loss is not None:
            losses.append(loss)

        if (ep + 1) % LOG_EVERY == 0:
            window  = min(LOG_EVERY, len(ep_rewards))
            avg_rew = float(np.mean(ep_rewards[-window:]))
            dist    = action_counts / max(1, action_counts.sum())
            dist_str = "  ".join(f"{ACTIONS[i]}={dist[i]:.2%}" for i in range(4))
            print(f"  ep {ep+1:>6,}/{EPISODES:,}  eps={agent.eps:.3f}  "
                  f"avg_reward={avg_rew:>8.1f}  [{dist_str}]")

    elapsed = time.perf_counter() - t0
    print(f"\nTraining done in {elapsed:.1f}s")

    # ── Save model ────────────────────────────────────────────────────────────
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    agent.save(OUT_MODEL)
    print(f"Model saved → {OUT_MODEL}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    final_dist = {ACTIONS[i]: int(action_counts[i]) for i in range(4)}
    last_k     = min(5_000, len(ep_rewards))
    metrics = {
        "episodes":               EPISODES,
        "elapsed_seconds":        round(elapsed, 1),
        "avg_reward_all":         round(float(np.mean(ep_rewards)), 2),
        "avg_reward_last_5k":     round(float(np.mean(ep_rewards[-last_k:])), 2),
        "action_counts":          final_dist,
        "action_distribution_pct":{ACTIONS[i]: round(float(action_counts[i]/action_counts.sum()*100), 1)
                                   for i in range(4)},
        "avg_loss_last_1k":       round(float(np.mean(losses[-1000:])) if losses else 0.0, 6),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved → {OUT_JSON}")

    # ── Reward curve plot ─────────────────────────────────────────────────────
    plot_reward_curve(ep_rewards, window=500, path=OUT_PLOT)
    print(f"Plot saved → {OUT_PLOT}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("═" * 52)
    print("  DQN v2 Training Summary")
    print("═" * 52)
    print(f"  Avg reward (all)       : ${metrics['avg_reward_all']:>10.1f}")
    print(f"  Avg reward (last 5k)   : ${metrics['avg_reward_last_5k']:>10.1f}")
    print(f"  Final epsilon          : {agent.eps:.4f}")
    print()
    print("  Action distribution:")
    for a, cnt in final_dist.items():
        pct = cnt / action_counts.sum() * 100
        print(f"    {a:<14} {cnt:>7,}  ({pct:.1f}%)")
    print("═" * 52)


if __name__ == "__main__":
    main()
