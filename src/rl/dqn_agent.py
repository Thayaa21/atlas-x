import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.optimization.threshold_optimizer import compute_customer_segments


COST_FN = 2000.0
COST_FP = 50.0
VIP_FRICTION_MULT = 10.0

ACTIONS = ["BLOCK", "FLAG", "APPROVE", "AUTO_APPROVE"]


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class DQNTrainingConfig:
    input_dim: int = 4
    output_dim: int = 4
    gamma: float = 0.0  # contextual bandit style: next_state irrelevant
    lr: float = 1e-3
    batch_size: int = 256
    buffer_size: int = 50_000
    train_steps: int = 100_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 80_000
    target_update_interval: int = 1_000


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idx = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


def encode_segment(segment: str) -> float:
    # VIP=1.0, Regular=0.5, New=0.0 after scaling
    if segment == "VIP":
        return 1.0
    if segment == "Regular":
        return 0.5
    return 0.0


def card_id_from_row(df: pd.DataFrame) -> pd.Series:
    return df["card1"].astype(str) + "-" + df["card2"].astype(str) + "-" + df["card3"].astype(str)


def compute_graph_risk_fallback(df: pd.DataFrame) -> np.ndarray:
    # For local/offline training we may not have Neo4j rings artifacts.
    # `cluster_fraud_rate` is already a graph-derived risk signal.
    if "cluster_fraud_rate" not in df.columns:
        raise ValueError("Expected `cluster_fraud_rate` column for graph_risk fallback.")
    return df["cluster_fraud_rate"].astype(float).to_numpy()


def compute_reward(y_true: np.ndarray, segments: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """
    Reward: -fraud_loss - (10 * VIP_friction)

    Action mapping:
      BLOCK, FLAG      => predict FRAUD (block/require review)
      APPROVE, AUTO_APPROVE => predict LEGIT (allow)
    """
    reward = np.zeros_like(y_true, dtype=np.float32)

    # Frauds missed: y_true=1 but allowed
    allowed = (actions == 2) | (actions == 3)  # APPROVE/AUTO_APPROVE
    missed = (y_true == 1) & allowed
    reward[missed] -= COST_FN

    # Legit falsely blocked: y_true=0 but flagged/blocked
    flagged = (actions == 0) | (actions == 1)  # BLOCK/FLAG
    false_alarm = (y_true == 0) & flagged
    reward[false_alarm] -= COST_FP

    # VIP friction for actions that require manual review
    vip_mask = segments == 1.0  # encoded VIP
    friction_actions = (actions == 0) | (actions == 1)
    vip_friction = vip_mask & friction_actions
    reward[vip_friction] -= VIP_FRICTION_MULT
    return reward


def train_dqn_offline(
    *,
    data_parquet: Path,
    model_path: Path,
    out_model_path: Path,
    out_dir: Path,
    episodes: int = 100_000,
    random_state: int = 42,
) -> Dict:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    cfg = DQNTrainingConfig(train_steps=episodes)

    df = pd.read_parquet(data_parquet)
    if "isFraud" not in df.columns:
        raise ValueError("Expected `isFraud` target column.")

    # Load base fraud model to compute fraud_prob feature.
    model = joblib.load(model_path)

    # Prepare classifier features (mirror training behavior in `train_v3_model.py`)
    X = df.drop(["isFraud", "TransactionID", "TransactionDT"], axis=1)
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].astype("object").fillna("None").astype("category")

    y = df["isFraud"].astype(int).to_numpy()
    fraud_prob = model.predict_proba(X)[:, 1].astype(np.float32)

    # Segment code feature
    seg_s, _cuts = compute_customer_segments(df)
    seg_enc = seg_s.map(encode_segment).astype(float).to_numpy().astype(np.float32)

    # Graph risk feature (offline fallback)
    graph_risk = compute_graph_risk_fallback(df).astype(np.float32)

    # Market context feature (simple deterministic proxy)
    # Using hour/day derived temporal signal: normalized hour in [0,1]
    market_context = (df["Transaction_Hour"].astype(float) / 23.0).to_numpy().astype(np.float32)

    # Build state vectors
    states = np.stack([fraud_prob, graph_risk, seg_enc, market_context], axis=1).astype(np.float32)

    # Replay buffer: contextual bandit -> next_state doesn't matter
    buffer = ReplayBuffer(capacity=cfg.buffer_size, state_dim=cfg.input_dim)

    policy_net = QNetwork(cfg.input_dim, cfg.output_dim)
    target_net = QNetwork(cfg.input_dim, cfg.output_dim)
    target_net.load_state_dict(policy_net.state_dict())

    opt = optim.Adam(policy_net.parameters(), lr=cfg.lr)

    steps = cfg.train_steps
    eps = cfg.epsilon_start
    eps_decay = (cfg.epsilon_start - cfg.epsilon_end) / max(1, cfg.epsilon_decay_steps)

    action_counts = np.zeros((cfg.output_dim,), dtype=np.int64)

    # Training loop
    for step in range(steps):
        idx = np.random.randint(0, len(df))
        state = states[idx]

        # Epsilon-greedy
        if np.random.rand() < eps:
            action = np.random.randint(0, cfg.output_dim)
        else:
            with torch.no_grad():
                q = policy_net(torch.from_numpy(state).unsqueeze(0))
                action = int(torch.argmax(q, dim=1).item())

        action_counts[action] += 1

        # Next state sample (unused because gamma=0.0, but kept for completeness)
        next_idx = np.random.randint(0, len(df))
        next_state = states[next_idx]

        reward = compute_reward(
            y_true=np.array([y[idx]], dtype=int),
            segments=np.array([seg_enc[idx]], dtype=float),
            actions=np.array([action], dtype=int),
        )[0]

        done = 0.0  # single-step contextual bandit
        buffer.add(state, action, float(reward), next_state, done)

        # Train when buffer has enough
        if buffer.size >= cfg.batch_size:
            s_b, a_b, r_b, ns_b, d_b = buffer.sample(cfg.batch_size)
            s_b_t = torch.from_numpy(s_b)
            a_b_t = torch.from_numpy(a_b)
            r_b_t = torch.from_numpy(r_b)

            # Q(s,a)
            q_sa = policy_net(s_b_t).gather(1, a_b_t.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                # gamma=0.0 -> target = r
                q_next = target_net(torch.from_numpy(ns_b)).max(dim=1).values
                target = r_b_t + cfg.gamma * q_next * (1.0 - torch.from_numpy(d_b))

            loss = nn.functional.mse_loss(q_sa, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Epsilon schedule
        eps = max(cfg.epsilon_end, eps - eps_decay)

        if (step + 1) % cfg.target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (step + 1) % 20_000 == 0:
            print(f"[DQN] step {step+1}/{steps} eps={eps:.3f}")

    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy_net.state_dict(),
            "input_dim": cfg.input_dim,
            "output_dim": cfg.output_dim,
            "actions": ACTIONS,
            "segment_coding": {"VIP": 1.0, "Regular": 0.5, "New": 0.0},
        },
        out_model_path,
    )

    # Export metadata for API/serving
    meta_path = out_dir / "dqn_metadata.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "episodes": episodes,
        "action_counts": {ACTIONS[i]: int(c) for i, c in enumerate(action_counts)},
        "action_space": ACTIONS,
        "state_features": ["fraud_prob", "graph_risk", "customer_segment", "market_context"],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"model_path": str(out_model_path), "meta_path": str(meta_path), "action_counts": meta["action_counts"]}


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    data_p = repo_root / "data/processed/train_full_features.parquet"
    fraud_model_p = repo_root / "src/models/atlass_x_xgb_v3.pkl"
    out_dir = repo_root / "src/rl"
    out_model = out_dir / "trained_dqn.pth"

    episodes = int(os.getenv("DQN_EPISODES", "100000"))
    res = train_dqn_offline(
        data_parquet=data_p,
        model_path=fraud_model_p,
        out_model_path=out_model,
        out_dir=out_dir,
        episodes=episodes,
    )
    print("Saved DQN:", res["model_path"])

