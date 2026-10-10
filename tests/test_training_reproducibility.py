import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


PROBE = r'''
import hashlib
import json
import os

import numpy as np

from agent.dqn_agent import DQNAgent
from common.reproducibility import set_global_seed
from config.dqn_hyperparameters import DQNHyperparameters


seed = int(os.environ["ASSURED_TEST_SEED"])


def weights_sha256(model):
    digest = hashlib.sha256()

    for weight in model.get_weights():
        array = np.asarray(weight)

        digest.update(
            str(array.shape).encode("utf-8")
        )
        digest.update(
            str(array.dtype).encode("utf-8")
        )
        digest.update(
            array.tobytes(order="C")
        )

    return digest.hexdigest()


set_global_seed(seed)

agent = DQNAgent(
    input_dim=6,
    action_dim=22,
    hidden_units=(64, 64),
    hyperparameters=DQNHyperparameters(
        replay_capacity=100,
        batch_size=2,
        min_replay_size=2,
        target_update_interval=2,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9,
    ),
    seed=seed,
)

initial_online = weights_sha256(
    agent.online_model
)

initial_target = weights_sha256(
    agent.target_model
)

losses = []

for i in range(6):
    state = np.asarray(
        [
            0.10 + 0.01 * i,
            0.20 + 0.01 * i,
            0.30 + 0.01 * i,
            0.40 + 0.01 * i,
            0.50 + 0.01 * i,
            0.60 + 0.01 * i,
        ],
        dtype=np.float32,
    )

    next_state = (
        state
        + np.float32(0.01)
    )

    agent.remember(
        state=state,
        action=i % 22,
        reward=float((i - 2) / 4.0),
        next_state=next_state,
        done=False,
    )

    loss = agent.train_step()

    if loss is not None:
        losses.append(float(loss))
        agent.decay_epsilon()


print(
    json.dumps(
        {
            "seed": seed,
            "initial_online_sha256":
                initial_online,
            "initial_target_sha256":
                initial_target,
            "final_online_sha256":
                weights_sha256(
                    agent.online_model
                ),
            "final_target_sha256":
                weights_sha256(
                    agent.target_model
                ),
            "training_steps":
                int(agent.training_steps),
            "epsilon":
                float(agent.epsilon),
            "losses":
                losses,
        },
        sort_keys=True,
    )
)
'''


def run_probe(seed):
    env = os.environ.copy()

    env["PYTHONHASHSEED"] = str(seed)
    env["ASSURED_TEST_SEED"] = str(seed)

    # Scientific training is currently validated
    # on the CPU execution path.
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"

    existing = env.get(
        "PYTHONPATH",
        "",
    )

    paths = [
        str(ROOT / "src"),
    ]

    if existing:
        paths.append(existing)

    env["PYTHONPATH"] = os.pathsep.join(
        paths
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            PROBE,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, (
        "training reproducibility subprocess "
        "failed:\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )

    return json.loads(
        completed.stdout.strip()
    )


def test_same_seed_reproduces_full_short_training():
    first = run_probe(7)
    second = run_probe(7)

    assert first == second

    assert (
        first["initial_online_sha256"]
        ==
        first["initial_target_sha256"]
    )

    assert first["training_steps"] == 5

    assert len(first["losses"]) == 5


def test_different_seed_changes_initial_network():
    first = run_probe(7)
    different = run_probe(17)

    assert (
        first["initial_online_sha256"]
        !=
        different[
            "initial_online_sha256"
        ]
    )
