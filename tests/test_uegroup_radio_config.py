import sys
from pathlib import Path

import simpy


ROOT = Path(__file__).resolve().parents[1]

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

sys.path.insert(
    0,
    str(ROOT / "simulator"),
)

sys.path.insert(
    0,
    str(ROOT / "simulator" / "lib"),
)



from Cell import Cell
from Results import UEgroup
from config.experiment_config import ExperimentConfig
from environment.radio_scenarios import STATIC_GOOD_DB


def test_uegroup_propagates_radio_scenario_to_ue():
    env = simpy.Environment()

    config = ExperimentConfig(
        model_variant="M2",
        seed=17,
        control_interval_ms=1,
        radio_scenario="STATIC_GOOD",
    )

    cell = Cell(
        "c1",
        [10],       # bandwidth
        "FR1",
        False,      # debug mode
        81920,      # UE buffer size
        False,      # FDD
        1,          # control interval
        "AQM2",
        experiment_config=config,
    )

    ue_group = UEgroup(
        2,              # DL users
        0,              # UL users
        300,            # DL packet size
        0,              # UL packet size
        0.6,            # DL packet arrival
        0,              # UL packet arrival
        0,              # activation time
        1,              # deactivation time
        "Constant",     # packet size distribution
        "Uniform",      # arrival distribution
        "URLLC",
        5,              # delay requirement
        "",
        "Rr",
        "SU",
        4,
        cell,
        100.0,          # simulation time
        1.0,            # measurement/radio update interval
        env,
        "S37",          # legacy initial SINR
        1.0,            # scheduling deadline
        experiment_config=cell.experiment_config,
    )

    assert cell.experiment_config is config

    assert len(
        ue_group.usersDL
    ) == 2

    for ue in ue_group.usersDL:
        assert (
            ue.radioLinks.radio_scenario
            is not None
        )

        assert (
            ue.radioLinks.linkQuality
            == STATIC_GOOD_DB
        )

        assert (
            ue.radioLinks.radio_scenario.scenario
            == "STATIC_GOOD"
        )

        assert (
            ue.radioLinks.radio_scenario.experiment_seed
            == 17
        )
