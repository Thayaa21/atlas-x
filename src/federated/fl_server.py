import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch

from flwr.common import ndarrays_to_parameters

from src.rl.dqn_agent import ACTIONS, QNetwork


repo_root = Path(__file__).resolve().parents[2]


def load_initial_dqn_state() -> List[np.ndarray]:
    ckpt = torch.load(repo_root / "src/rl/trained_dqn.pth", map_location="cpu")
    input_dim = int(ckpt["input_dim"])
    output_dim = int(ckpt["output_dim"])
    model = QNetwork(input_dim, output_dim)
    model.load_state_dict(ckpt["state_dict"])
    # Keep ordering consistent with `model.state_dict().values()`
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def main() -> None:
    host_port = os.getenv("FL_SERVER_ADDRESS", "0.0.0.0:8080")
    rounds = int(os.getenv("FL_ROUNDS", "3"))
    fraction_fit = float(os.getenv("FL_FRACTION_FIT", "1.0"))
    min_fit_clients = int(os.getenv("FL_MIN_FIT_CLIENTS", "5"))

    input_parameters = load_initial_dqn_state()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_available_clients=5,
        evaluate_fn=None,
        initial_parameters=ndarrays_to_parameters(input_parameters),
    )

    fl.server.start_server(
        server_address=host_port,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

