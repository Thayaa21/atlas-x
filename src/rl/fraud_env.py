"""
ATLAS-X Fraud Detection Environment
─────────────────────────────────────
Wraps the 20% holdout as a contextual-bandit environment for DQN training.

Each "episode" is a single transaction:
  state  = [fraud_prob, segment_encoded, graph_risk_score, market_volatility]
  action = APPROVE / FLAG / BLOCK / AUTO_APPROVE
  reward = business-cost signal (see REWARD TABLE below)

REWARD TABLE
─────────────────────────────────────────────
Action          True label  Segment   Reward
─────────────────────────────────────────────
APPROVE         fraud       any       -2000
APPROVE         legit       any       +10
AUTO_APPROVE    fraud       any       -2000
AUTO_APPROVE    legit       any       +15   (slightly better; no latency cost)
BLOCK           fraud       any       +100
BLOCK           legit       VIP       -5000
BLOCK           legit       Regular   -50
BLOCK           legit       New       -50
FLAG            fraud       any       +50   (caught but slower)
FLAG            legit       VIP       -100  (VIP review friction > Regular)
FLAG            legit       Regular   -25
FLAG            legit       New       -25
─────────────────────────────────────────────
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.rl.dqn_agent_v2 import (
    APPROVE, AUTO_APPROVE, BLOCK, FLAG,
    STATE_DIM, encode_segment,
)

# ── Reward constants ──────────────────────────────────────────────────────────
R_APPROVE_FRAUD     = -2000.0
R_APPROVE_LEGIT     =   +10.0
R_AUTO_APPROVE_FRAUD= -2000.0
R_AUTO_APPROVE_LEGIT=   +15.0
R_BLOCK_FRAUD       =  +100.0
R_BLOCK_LEGIT_VIP   = -5000.0
R_BLOCK_LEGIT_OTHER =   -50.0
R_FLAG_FRAUD        =   +50.0
R_FLAG_LEGIT_VIP    =  -100.0
R_FLAG_LEGIT_OTHER  =   -25.0


def compute_reward(action: int, is_fraud: bool, segment: str) -> float:
    if action == APPROVE:
        return R_APPROVE_FRAUD if is_fraud else R_APPROVE_LEGIT
    if action == AUTO_APPROVE:
        return R_AUTO_APPROVE_FRAUD if is_fraud else R_AUTO_APPROVE_LEGIT
    if action == BLOCK:
        if is_fraud:
            return R_BLOCK_FRAUD
        return R_BLOCK_LEGIT_VIP if segment == "VIP" else R_BLOCK_LEGIT_OTHER
    if action == FLAG:
        if is_fraud:
            return R_FLAG_FRAUD
        return R_FLAG_LEGIT_VIP if segment == "VIP" else R_FLAG_LEGIT_OTHER
    raise ValueError(f"Unknown action: {action}")


class FraudEnv:
    """
    Stateless contextual-bandit environment over a transaction dataset.

    Parameters
    ----------
    fraud_probs     : 1-D float array, v4 model output for each row
    graph_risks     : 1-D float array, graph_risk_score for each row
    segments        : 1-D str array,   customer segment label
    labels          : 1-D int array,   ground truth isFraud
    market_vol      : 1-D float array | None  (defaults to Transaction_Hour/23)
    seed            : RNG seed
    """

    def __init__(
        self,
        fraud_probs:  np.ndarray,
        graph_risks:  np.ndarray,
        segments:     np.ndarray,
        labels:       np.ndarray,
        market_vol:   Optional[np.ndarray] = None,
        seed:         int = 42,
    ):
        self.fraud_probs = fraud_probs.astype(np.float32)
        self.graph_risks = graph_risks.astype(np.float32)
        self.labels      = labels.astype(int)
        self.segments    = segments                            # str array
        self.seg_enc     = np.array([encode_segment(s) for s in segments],
                                    dtype=np.float32)
        self.market_vol  = (
            market_vol.astype(np.float32)
            if market_vol is not None
            else np.full(len(labels), 0.5, dtype=np.float32)
        )
        self.n           = len(labels)
        self.rng         = np.random.default_rng(seed)
        self._cur_idx: Optional[int] = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _state_at(self, idx: int) -> np.ndarray:
        return np.array([
            self.fraud_probs[idx],
            self.seg_enc[idx],
            self.graph_risks[idx],
            self.market_vol[idx],
        ], dtype=np.float32)

    # ── OpenAI Gym-style API ──────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Sample a random transaction and return its state."""
        self._cur_idx = int(self.rng.integers(0, self.n))
        return self._state_at(self._cur_idx)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply action to current transaction.

        Returns
        -------
        next_state  : state of next randomly sampled transaction
        reward      : business reward for this decision
        done        : always True (single-step bandit)
        info        : diagnostic dict
        """
        idx      = self._cur_idx
        is_fraud = bool(self.labels[idx])
        segment  = str(self.segments[idx])
        reward   = compute_reward(action, is_fraud, segment)

        # Sample next transaction for the replay-buffer next_state field
        next_idx        = int(self.rng.integers(0, self.n))
        self._cur_idx   = next_idx
        next_state      = self._state_at(next_idx)

        info = {
            "idx":      idx,
            "is_fraud": is_fraud,
            "segment":  segment,
            "action":   action,
        }
        return next_state, reward, True, info

    # ── Batch utility (for offline evaluation) ───────────────────────────────

    def all_states(self) -> np.ndarray:
        """Return the full (N, STATE_DIM) state matrix."""
        return np.stack([
            self.fraud_probs,
            self.seg_enc,
            self.graph_risks,
            self.market_vol,
        ], axis=1)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        fraud_probs: np.ndarray,
        *,
        seed: int = 42,
    ) -> "FraudEnv":
        """
        Build env from a DataFrame that has isFraud, customer_segment, and
        graph_risk_score, plus optionally Transaction_Hour.
        """
        from src.optimization.threshold_optimizer import compute_customer_segments

        if "customer_segment" not in df.columns:
            seg_s, _ = compute_customer_segments(df)
            segments = seg_s.to_numpy()
        else:
            segments = df["customer_segment"].to_numpy()

        graph_risks = df["graph_risk_score"].fillna(0.0).to_numpy().astype(np.float32)
        labels      = df["isFraud"].astype(int).to_numpy()

        if "Transaction_Hour" in df.columns:
            market_vol = (df["Transaction_Hour"].astype(float) / 23.0).to_numpy().astype(np.float32)
        else:
            market_vol = None

        return cls(
            fraud_probs=fraud_probs,
            graph_risks=graph_risks,
            segments=segments,
            labels=labels,
            market_vol=market_vol,
            seed=seed,
        )
