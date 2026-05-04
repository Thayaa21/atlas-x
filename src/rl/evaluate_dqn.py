"""
ATLAS-X – DQN v2 Evaluation
─────────────────────────────
Compares three decision strategies on the same 20% holdout:

  Strategy 1 – Static thresholds (v3 model)
      VIP threshold=0.90, Regular=0.60, New=0.40
      Actions: APPROVE (below) or BLOCK (above)

  Strategy 2 – Static thresholds + graph (v4 model)
      Same threshold values, but fraud_prob comes from the
      graph-feature-augmented v4 model.

  Strategy 3 – DQN v2 agent
      4-way action space (APPROVE / FLAG / BLOCK / AUTO_APPROVE)
      driven by [fraud_prob, segment, graph_risk, market_vol]

All three are evaluated on identical holdout rows with the same
reward function used during DQN training (see fraud_env.py).

Saves:
  results/dqn_comparison.json
  results/dqn_comparison.png

Usage:
    python -m src.rl.evaluate_dqn
"""
import json
from pathlib import Path
from typing import NamedTuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.optimization.threshold_optimizer import compute_customer_segments
from src.rl.dqn_agent_v2 import ACTIONS, APPROVE, AUTO_APPROVE, BLOCK, FLAG, DQNAgentV2
from src.rl.fraud_env import FraudEnv, compute_reward

REPO      = Path(__file__).resolve().parents[2]
V3_MODEL  = REPO / "src/models/atlass_x_xgb_v3.pkl"
V4_MODEL  = REPO / "src/models/atlass_x_xgb_v4_graph.pkl"
DQN_MODEL = REPO / "src/rl/trained_dqn_v2.pth"
V3_DATA   = REPO / "data/processed/train_full_features.parquet"
V4_DATA   = REPO / "data/processed/train_with_graph_features.parquet"
THRESH_P  = REPO / "src/optimization/artifacts/thresholds.json"
OUT_JSON  = REPO / "results/dqn_comparison.json"
OUT_PLOT  = REPO / "results/dqn_comparison.png"

GRAPH_COLS = ["device_fraud_rate", "device_card_velocity", "connected_fraud_cards",
              "email_fraud_rate", "address_fraud_rate", "graph_risk_score"]
DROP_COLS  = ["isFraud", "TransactionID", "TransactionDT", "customer_segment"]


# ── Data loading ──────────────────────────────────────────────────────────────

class HoldoutData(NamedTuple):
    X_test:     pd.DataFrame
    y_test:     np.ndarray
    seg_test:   np.ndarray     # str labels
    df_test:    pd.DataFrame   # original rows with all columns


def _load_holdout(data_path: Path) -> HoldoutData:
    from sklearn.model_selection import StratifiedShuffleSplit

    df = pd.read_parquet(data_path)
    seg_s, _ = compute_customer_segments(df)
    df = df.copy()
    df["customer_segment"] = seg_s.values

    X = df.drop([c for c in DROP_COLS if c in df.columns], axis=1)
    y = df["isFraud"].astype(int).to_numpy()

    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].astype("object").fillna("None").astype("category")
    for col in GRAPH_COLS:
        if col in X.columns:
            X[col] = X[col].fillna(0.0)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(sss.split(X, y))

    return HoldoutData(
        X_test=X.iloc[test_idx].reset_index(drop=True),
        y_test=y[test_idx],
        seg_test=df["customer_segment"].iloc[test_idx].reset_index(drop=True).to_numpy(),
        df_test=df.iloc[test_idx].reset_index(drop=True),
    )


# ── Reward & metrics helpers ──────────────────────────────────────────────────

def _total_reward(actions: np.ndarray, y_true: np.ndarray,
                  segments: np.ndarray) -> float:
    return float(sum(
        compute_reward(int(a), bool(y), str(s))
        for a, y, s in zip(actions, y_true, segments)
    ))


def _strategy_metrics(actions: np.ndarray, y_true: np.ndarray,
                      segments: np.ndarray, label: str) -> dict:
    # For binary recall/precision: FLAG + BLOCK = predict fraud; APPROVE + AUTO = predict legit
    y_pred = np.where(np.isin(actions, [FLAG, BLOCK]), 1, 0)
    total_rew = _total_reward(actions, y_true, segments)
    action_dist = {ACTIONS[i]: int((actions == i).sum()) for i in range(4)}

    return {
        "strategy":    label,
        "total_reward": round(total_rew, 0),
        "avg_reward":   round(total_rew / len(y_true), 4),
        "recall":       round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision":    round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1":           round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "action_counts": action_dist,
    }


# ── Strategy 1: v3 static thresholds ─────────────────────────────────────────

def strategy_v3(hd: HoldoutData, thresholds: dict) -> np.ndarray:
    model      = joblib.load(V3_MODEL)
    fraud_prob = model.predict_proba(hd.X_test)[:, 1]
    actions    = np.full(len(fraud_prob), APPROVE, dtype=int)
    for seg, thr in thresholds.items():
        mask = hd.seg_test == seg
        actions[mask] = np.where(fraud_prob[mask] >= thr, BLOCK, APPROVE)
    return actions


# ── Strategy 2: v4 static thresholds + graph ─────────────────────────────────

def strategy_v4(hd: HoldoutData, thresholds: dict) -> np.ndarray:
    model      = joblib.load(V4_MODEL)
    fraud_prob = model.predict_proba(hd.X_test)[:, 1]
    actions    = np.full(len(fraud_prob), APPROVE, dtype=int)
    for seg, thr in thresholds.items():
        mask = hd.seg_test == seg
        actions[mask] = np.where(fraud_prob[mask] >= thr, BLOCK, APPROVE)
    return actions


# ── Strategy 3: DQN v2 ────────────────────────────────────────────────────────

def strategy_dqn(hd: HoldoutData, fraud_prob_v4: np.ndarray) -> np.ndarray:
    agent = DQNAgentV2.load(DQN_MODEL)
    env   = FraudEnv.from_dataframe(
        hd.df_test.assign(customer_segment=hd.seg_test),
        fraud_prob_v4,
        seed=42,
    )
    states  = env.all_states()
    actions = np.array([agent.select_action(s, greedy=True) for s in states], dtype=int)
    return actions


# ── Comparison plot ───────────────────────────────────────────────────────────

def _plot(results: list[dict], path: Path) -> None:
    labels   = [r["strategy"] for r in results]
    metrics  = ["recall", "precision", "f1"]
    x        = np.arange(len(metrics))
    width    = 0.25
    colors   = ["#4C72B0", "#DD8452", "#55A868"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: recall / precision / F1
    for i, (r, c) in enumerate(zip(results, colors)):
        vals = [r[m] for m in metrics]
        bars = ax1.bar(x + i * width, vals, width, label=r["strategy"], color=c, alpha=0.85)
        for b in bars:
            ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                     f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(["Recall", "Precision", "F1"])
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("Score")
    ax1.set_title("Detection Metrics")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Right: total reward (bar per strategy)
    rewards = [r["total_reward"] for r in results]
    bars2   = ax2.bar(labels, rewards, color=colors, alpha=0.85)
    for b, v in zip(bars2, rewards):
        ax2.text(b.get_x() + b.get_width()/2,
                 v + abs(min(rewards)) * 0.02,
                 f"${v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Total Reward ($)")
    ax2.set_title("Total Business Reward (higher = better)")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("ATLAS-X  Strategy Comparison  –  20% Holdout", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not DQN_MODEL.exists():
        raise FileNotFoundError(
            f"{DQN_MODEL} not found.\n"
            "Run  python -m src.rl.train_dqn_v2  first."
        )

    thresh_art = json.loads(THRESH_P.read_text())
    thresholds = {s: float(thresh_art["segments"][s]["threshold"]) for s in ["VIP", "Regular", "New"]}

    # ── Load holdouts (separate for v3 / v4 feature sets) ─────────────────
    print("Loading v3 holdout...")
    hd_v3  = _load_holdout(V3_DATA)
    print("Loading v4 holdout (with graph features)...")
    hd_v4  = _load_holdout(V4_DATA)

    # Precompute v4 fraud_prob (needed by both strategy 2 and DQN)
    print("Running v4 model...")
    v4_model      = joblib.load(V4_MODEL)
    fraud_prob_v4 = v4_model.predict_proba(hd_v4.X_test)[:, 1].astype(np.float32)

    # ── Run strategies ────────────────────────────────────────────────────
    print("Strategy 1: v3 static thresholds...")
    act_v3  = strategy_v3(hd_v3, thresholds)

    print("Strategy 2: v4 static thresholds + graph features...")
    act_v4  = strategy_v4(hd_v4, thresholds)

    print("Strategy 3: DQN v2 agent...")
    act_dqn = strategy_dqn(hd_v4, fraud_prob_v4)

    # ── Metrics ───────────────────────────────────────────────────────────
    m1 = _strategy_metrics(act_v3,  hd_v3.y_test, hd_v3.seg_test, "v3 static")
    m2 = _strategy_metrics(act_v4,  hd_v4.y_test, hd_v4.seg_test, "v4 static+graph")
    m3 = _strategy_metrics(act_dqn, hd_v4.y_test, hd_v4.seg_test, "DQN v2")
    results = [m1, m2, m3]

    # ── Console table ─────────────────────────────────────────────────────
    print()
    print("═" * 72)
    print("  STRATEGY COMPARISON  –  20% holdout")
    print("═" * 72)
    print(f"  {'Strategy':<22} {'Recall':>8} {'Prec':>8} {'F1':>8} {'Avg Reward':>12} {'Total Reward':>14}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*14}")
    for r in results:
        print(f"  {r['strategy']:<22} {r['recall']:>8.2%} {r['precision']:>8.2%} "
              f"{r['f1']:>8.2%} {r['avg_reward']:>12.2f} ${r['total_reward']:>13,.0f}")
    print("═" * 72)

    print("\nDQN action distribution:")
    for a, cnt in m3["action_counts"].items():
        pct = cnt / sum(m3["action_counts"].values()) * 100
        print(f"  {a:<14} {cnt:>7,}  ({pct:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────────────
    output = {
        "holdout_size":  int(len(hd_v4.y_test)),
        "total_fraud":   int(hd_v4.y_test.sum()),
        "thresholds":    thresholds,
        "strategies":    results,
        "delta_dqn_vs_v3": {
            "recall":       round(m3["recall"]    - m1["recall"],    4),
            "precision":    round(m3["precision"] - m1["precision"], 4),
            "f1":           round(m3["f1"]        - m1["f1"],        4),
            "total_reward": round(m3["total_reward"] - m1["total_reward"], 0),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"\nSaved → results/dqn_comparison.json")

    _plot(results, OUT_PLOT)
    print(f"Saved → results/dqn_comparison.png")


if __name__ == "__main__":
    main()
