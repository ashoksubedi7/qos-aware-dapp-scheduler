import sys
from pathlib import Path

import simpy


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

sys.path.insert(
    0,
    str(ROOT / "simulator"),
)


from UE import RadioLink
from config.experiment_config import ExperimentConfig
from environment.radio_scenarios import (
    STATIC_GOOD_DB,
)


def test_legacy_radio_link_has_no_assured_scenario():
    radio = RadioLink(
        1,
        37.0,
        "ue1",
    )

    assert radio.radio_scenario is None
    assert radio.linkQuality == 37.0


def test_assured_static_good_overrides_legacy_initial_sinr():
    config = ExperimentConfig(
        model_variant="M2",
        seed=17,
        radio_scenario="STATIC_GOOD",
    )

    radio = RadioLink(
        1,
        37.0,
        "ue1",
        experiment_config=config,
        slice_name="URLLC",
        t_sim=1000,
        update_interval=50,
    )

    assert radio.radio_scenario is not None
    assert radio.linkQuality == STATIC_GOOD_DB


def test_same_assured_radio_configuration_replays_same_trace():
    config = ExperimentConfig(
        model_variant="M2",
        seed=17,
        radio_scenario="FLUCTUATING",
    )

    radio_a = RadioLink(
        1,
        37.0,
        "ue3",
        experiment_config=config,
        slice_name="URLLC",
        t_sim=1000,
        update_interval=50,
    )

    radio_b = RadioLink(
        1,
        37.0,
        "ue3",
        experiment_config=config,
        slice_name="URLLC",
        t_sim=1000,
        update_interval=50,
    )

    trace_a = [
        radio_a.radio_scenario.sinr_at(t)
        for t in range(
            50,
            500,
            50,
        )
    ]

    trace_b = [
        radio_b.radio_scenario.sinr_at(t)
        for t in range(
            50,
            500,
            50,
        )
    ]

    assert trace_a == trace_b


def test_updateLQ_uses_assured_radio_scenario():
    config = ExperimentConfig(
        model_variant="M2",
        seed=7,
        radio_scenario="STATIC_GOOD",
    )

    radio = RadioLink(
        1,
        37.0,
        "ue1",
        experiment_config=config,
        slice_name="eMBB",
        t_sim=1000,
        update_interval=50,
    )

    env = simpy.Environment()

    env.process(
        radio.updateLQ(
            env,
            udIntrv=50,
            tSim=1000,
            fl=False,
            u=1,
            r="",
        )
    )

    env.run(
        until=51
    )

    assert radio.linkQuality == STATIC_GOOD_DB
