"""
ATLAS-X DQN Agent v2
─────────────────────
Contextual-bandit Deep Q-Network for fraud action selection.

State (4 features):
  [fraud_prob, segment_encoded, graph_risk_score, market_volatility]

Actions:
  APPROVE=0      – pass immediately
  FLAG=1         – route to manual review
  BLOCK=2        – decline transaction
  AUTO_APPROVE=3 – instant pass for verified low-risk

Network: Linear(4→128) → ReLU → Linear(128→64) → ReLU → Linear(64→4)

Vs v1 changes:
  - Wider network (128 hidden vs 64)
  - graph_risk_score from v4 parquet (not cluster_fraud_rate fallback)
  - Action ordering matches task spec (APPROVE=0, not BLOCK=0)
  - Clean agent class with select_action / push_experience / train_step
"""
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Action constants ──────────────────────────────────────────────────────────
APPROVE      = 0
FLAG         = 1
BLOCK        = 2
AUTO_APPROVE = 3
ACTIONS      = ["APPROVE", "FLAG", "BLOCK", "AUTO_APPROVE"]
N_ACTIONS    = 4
STATE_DIM    = 4


# ── Segment encoder ───────────────────────────────────────────────────────────

def encode_segment(seg: str) -> float:
    return {"VIP": 1.0, "Regular": 0.5}.get(seg, 0.0)


# ── Network ───────────────────────────────────────────────────────────────────

class QNetworkV2(nn.Module):
    """Input→128→64→N_ACTIONS."""

    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int = STATE_DIM):
        self.capacity  = capacity
        self.state_dim = state_dim
        self._buf: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: float) -> None:
        self._buf.append((state.copy(), action, reward, next_state.copy(), done))

    def sample(self, batch_size: int):
        idx  = np.random.choice(len(self._buf), size=batch_size, replace=False)
        batch = [self._buf[i] for i in idx]
        s, a, r, ns, d = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.int64),
                np.array(r,  dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d,  dtype=np.float32))

    def __len__(self) -> int:
        return len(self._buf)


# ── Agent ─────────────────────────────────────────────────────────────────────

class DQNAgentV2:
    """
    Contextual-bandit DQN (gamma=0): no temporal dependencies between
    fraud decisions, so discount factor is 0 and next-state is unused.
    """

    def __init__(
        self,
        state_dim:   int   = STATE_DIM,
        n_actions:   int   = N_ACTIONS,
        lr:          float = 1e-3,
        gamma:       float = 0.0,      # contextual bandit
        buffer_size: int   = 10_000,
        batch_size:  int   = 64,
        eps_start:   float = 1.0,
        eps_end:     float = 0.01,
        eps_decay_steps: int = 40_000,
        target_update_every: int = 1_000,
        device: Optional[str] = None,
    ):
        self.gamma      = gamma
        self.batch_size = batch_size
        self.eps        = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = (eps_start - eps_end) / max(1, eps_decay_steps)
        self.target_update_every = target_update_every
        self._step      = 0

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy_net = QNetworkV2(state_dim, n_actions).to(self.device)
        self.target_net = QNetworkV2(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer     = ReplayBuffer(buffer_size, state_dim)

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.eps:
            return int(np.random.randint(0, N_ACTIONS))
        with torch.no_grad():
            t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
            return int(self.policy_net(t).argmax(dim=1).item())

    # ── Experience storage ────────────────────────────────────────────────────

    def push(self, state, action, reward, next_state, done=0.0) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s_t  = torch.from_numpy(s).to(self.device)
        a_t  = torch.from_numpy(a).to(self.device)
        r_t  = torch.from_numpy(r).to(self.device)
        ns_t = torch.from_numpy(ns).to(self.device)
        d_t  = torch.from_numpy(d).to(self.device)

        # Q(s, a)
        q_sa = self.policy_net(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            q_next  = self.target_net(ns_t).max(dim=1).values
            target  = r_t + self.gamma * q_next * (1.0 - d_t)

        loss = nn.functional.mse_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay & target sync
        self.eps = max(self.eps_end, self.eps - self.eps_decay)
        self._step += 1
        if self._step % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path) -> None:
        torch.save({
            "state_dict": self.policy_net.state_dict(),
            "state_dim":  STATE_DIM,
            "n_actions":  N_ACTIONS,
            "actions":    ACTIONS,
            "eps":        self.eps,
        }, path)

    @classmethod
    def load(cls, path, **kwargs) -> "DQNAgentV2":
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(**kwargs)
        agent.policy_net.load_state_dict(ckpt["state_dict"])
        agent.policy_net.eval()
        return agent
