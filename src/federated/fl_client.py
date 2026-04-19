import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import flwr as fl
import torch

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from src.rl.dqn_agent import ACTIONS, QNetwork, compute_reward, encode_segment
from src.optimization.threshold_optimizer import compute_customer_segments


repo_root = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def segment_params_from_thresholds() -> Dict[str, float]:
    th = _load_json(repo_root / "src/optimization/artifacts/thresholds.json")
    return th["segment_quantile_cuts"]


def segment_from_cuts_batch(d1: np.ndarray, d15: np.ndarray, seg_params: Dict[str, float]) -> np.ndarray:
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

    seg = np.where(combined >= vip_cut, "VIP", np.where(combined >= regular_cut, "Regular", "New"))
    enc = np.vectorize(lambda s: encode_segment(s))(seg).astype(np.float32)
    return enc


class DQNClient(fl.client.NumPyClient):
    def __init__(self, *, bank_id: int, local_steps: int):
        self.bank_id = bank_id
        self.local_steps = local_steps

        self.v3_model = joblib.load(repo_root / "src/models/atlass_x_xgb_v3.pkl")
        self.seg_params = segment_params_from_thresholds()

        # Load local data partition.
        df = pd.read_parquet(repo_root / "data/processed/train_full_features.parquet")
        # Partition proxy: deterministic hash by TransactionID.
        part_mask = (df["TransactionID"].astype(int) % 5) == bank_id
        df = df.loc[part_mask].reset_index(drop=True)
        self.df = df

        # Precompute state vectors (fraud_prob, graph_risk, segment_enc, market_context)
        X = df.drop(["isFraud", "TransactionID", "TransactionDT"], axis=1)
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].astype("object").fillna("None").astype("category")
        y = df["isFraud"].astype(int).to_numpy()

        fraud_prob = self.v3_model.predict_proba(X)[:, 1].astype(np.float32)
        graph_risk = df["cluster_fraud_rate"].astype(float).to_numpy().astype(np.float32)
        market_context = (df["Transaction_Hour"].astype(float) / 23.0).to_numpy().astype(np.float32)

        seg_enc = segment_from_cuts_batch(
            d1=df["D1"].astype(float).to_numpy(),
            d15=df["D15"].astype(float).to_numpy(),
            seg_params=self.seg_params,
        )

        self.y = y
        self.states = np.stack([fraud_prob, graph_risk, seg_enc, market_context], axis=1).astype(np.float32)

        self.input_dim = self.states.shape[1]
        self.output_dim = len(ACTIONS)

    def get_parameters(self, config):
        # Initialize model with random weights; server parameters will be provided in fit.
        model = QNetwork(self.input_dim, self.output_dim)
        return [v.cpu().numpy() for v in model.state_dict().values()]

    def set_parameters(self, model: QNetwork, parameters):
        state_dict = model.state_dict()
        keys = list(state_dict.keys())
        # Flower provides ndarrays in the same order as parameters_to_ndarrays; we use model.state_dict ordering.
        for i, k in enumerate(keys):
            state_dict[k] = torch.tensor(parameters[i])
        model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        model = QNetwork(self.input_dim, self.output_dim)
        self.set_parameters(model, parameters)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch_size = int(config.get("batch_size", 256))
        epsilon = float(config.get("epsilon", 0.2))

        # DQN contextual bandit training (gamma=0.0): target = reward.
        states = self.states
        y = self.y

        for _ in range(self.local_steps):
            idx = np.random.randint(0, len(states))
            state = states[idx]
            # epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(0, self.output_dim)
            else:
                with torch.no_grad():
                    q = model(torch.from_numpy(state).unsqueeze(0))
                    action = int(torch.argmax(q, dim=1).item())

            # Reward computed from ground truth and action.
            reward = compute_reward(
                y_true=np.array([y[idx]], dtype=int),
                segments=np.array([state[2]], dtype=float),
                actions=np.array([action], dtype=int),
            )[0]

            # Supervise Q(s,a) to match reward.
            state_t = torch.from_numpy(state).unsqueeze(0)
            q_all = model(state_t)
            q_sa = q_all[0, action]
            loss = (q_sa - torch.tensor(reward, dtype=torch.float32)) ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        new_params = [v.cpu().numpy() for v in model.state_dict().values()]
        num_examples = len(states)
        metrics = {"local_steps": self.local_steps, "bank_id": self.bank_id, "num_examples": num_examples}
        return new_params, num_examples, metrics

    def evaluate(self, parameters, config):
        # Optional: could compute action distribution. Keep as stub.
        return 0.0, 0, {"bank_id": self.bank_id}


def main():
    bank_id = int(os.getenv("BANK_ID", "0"))
    local_steps = int(os.getenv("LOCAL_STEPS", "2000"))

    client = DQNClient(bank_id=bank_id, local_steps=local_steps)
    fl.client.start_numpy_client(server_address=os.getenv("FL_SERVER_ADDRESS", "127.0.0.1:8080"), client=client)


if __name__ == "__main__":
    main()

