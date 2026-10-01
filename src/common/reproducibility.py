import os
import random

import numpy as np
import tensorflow as tf


def set_global_seed(seed):
    """
    Set reproducible seeds for the main random-number sources used
    by AssuredQoS experiments.
    """
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    return seed
