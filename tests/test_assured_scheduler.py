import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "simulator"))
sys.path.insert(0, str(ROOT / "simulator" / "lib"))

from assured_scheduler import AssuredScheduler


def test_valid_model_variants():
    assert AssuredScheduler.VALID_VARIANTS == (
        "M1",
        "M2",
        "M3",
    )


def test_m1_definition():
    definition = {
        "input_dim": 6,
    }

    assert definition["input_dim"] == 6


def test_m2_definition():
    definition = {
        "input_dim": 7,
    }

    assert definition["input_dim"] == 7
