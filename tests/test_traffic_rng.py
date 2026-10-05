import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

sys.path.insert(
    0,
    str(ROOT / "simulator"),
)


from UE import PacketFlow

from config.experiment_config import (
    ExperimentConfig,
)


def make_flow(
    ue="ue1",
    seed=17,
):
    config = ExperimentConfig(
        seed=seed,
    )

    return PacketFlow(
        1,
        1200,
        1.2,
        ue,
        "DL",
        "eMBB",
        0,
        1,
        "Pareto",
        "Uniform",
        None,
        experiment_config=config,
    )


def test_same_flow_seed_reproduces_sequence():
    flow_a = make_flow()
    flow_b = make_flow()

    sizes_a = [
        flow_a.getPsize()
        for _ in range(20)
    ]

    sizes_b = [
        flow_b.getPsize()
        for _ in range(20)
    ]

    arrivals_a = [
        flow_a.getParrRate()
        for _ in range(20)
    ]

    arrivals_b = [
        flow_b.getParrRate()
        for _ in range(20)
    ]

    assert sizes_a == sizes_b

    assert np.allclose(
        arrivals_a,
        arrivals_b,
    )


def test_different_ues_have_different_streams():
    flow_a = make_flow(
        ue="ue1"
    )

    flow_b = make_flow(
        ue="ue2"
    )

    sequence_a = [
        flow_a.getPsize()
        for _ in range(20)
    ]

    sequence_b = [
        flow_b.getPsize()
        for _ in range(20)
    ]

    assert sequence_a != sequence_b


def test_global_random_consumption_does_not_change_flow():
    import random

    flow_a = make_flow()

    sequence_a = [
        flow_a.getPsize()
        for _ in range(20)
    ]

    # Deliberately disturb global RNGs.
    for _ in range(1000):
        random.random()

    np.random.random(
        1000
    )

    flow_b = make_flow()

    sequence_b = [
        flow_b.getPsize()
        for _ in range(20)
    ]

    assert sequence_a == sequence_b


def test_different_experiment_seeds_change_trace():
    flow_a = make_flow(
        seed=17
    )

    flow_b = make_flow(
        seed=18
    )

    sequence_a = [
        flow_a.getPsize()
        for _ in range(20)
    ]

    sequence_b = [
        flow_b.getPsize()
        for _ in range(20)
    ]

    assert sequence_a != sequence_b
