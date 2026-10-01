import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.reproducibility import (
    set_global_seed,
)


def test_seed_reproduces_python_random():
    set_global_seed(7)
    first = [
        random.random()
        for _ in range(5)
    ]

    set_global_seed(7)
    second = [
        random.random()
        for _ in range(5)
    ]

    assert first == second


def test_seed_reproduces_numpy():
    set_global_seed(11)
    first = np.random.random(5)

    set_global_seed(11)
    second = np.random.random(5)

    assert np.allclose(
        first,
        second,
    )


def test_seed_reproduces_tensorflow():
    set_global_seed(13)

    first = tf.random.uniform(
        shape=(5,),
    ).numpy()

    set_global_seed(13)

    second = tf.random.uniform(
        shape=(5,),
    ).numpy()

    assert np.allclose(
        first,
        second,
    )
