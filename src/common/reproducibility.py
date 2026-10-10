import os
import random

import numpy as np
import tensorflow as tf


def set_global_seed(seed):
    """
    Set the reproducibility controls used by
    AssuredQoS experiments.

    Scientific training processes must also be
    launched with PYTHONHASHSEED set before Python
    starts. Assigning PYTHONHASHSEED here records the
    requested value for child processes but cannot
    retroactively change hash randomization in the
    current interpreter.
    """
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    enable_determinism = getattr(
        tf.config.experimental,
        "enable_op_determinism",
        None,
    )

    if enable_determinism is not None:
        enable_determinism()

    return seed
