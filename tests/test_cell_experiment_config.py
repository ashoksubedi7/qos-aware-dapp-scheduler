import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

sys.path.insert(
    0,
    str(ROOT / "simulator"),
)


from Cell import Cell
from config.experiment_config import (
    ExperimentConfig,
)


def test_cell_uses_supplied_experiment_config():
    config = ExperimentConfig(
        model_variant="M2",
        seed=17,
        control_interval_ms=1,
    )

    cell = Cell(
        "c1",
        [10],
        "FR1",
        False,
        81920,
        False,
        1,
        "AQM2",
        experiment_config=config,
    )

    assert (
        cell.experiment_config
        is config
    )

    assert (
        cell.interSliceSched.config
        is config
    )


def test_cell_rejects_variant_mismatch():
    config = ExperimentConfig(
        model_variant="M1",
        control_interval_ms=1,
    )

    with pytest.raises(
        ValueError,
        match="scheduler/config mismatch",
    ):
        Cell(
            "c1",
            [10],
            "FR1",
            False,
            81920,
            False,
            1,
            "AQM2",
            experiment_config=config,
        )


def test_cell_rejects_interval_mismatch():
    config = ExperimentConfig(
        model_variant="M2",
        control_interval_ms=5,
    )

    with pytest.raises(
        ValueError,
        match="control interval",
    ):
        Cell(
            "c1",
            [10],
            "FR1",
            False,
            81920,
            False,
            1,
            "AQM2",
            experiment_config=config,
        )
