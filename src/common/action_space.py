import numpy as np


SLICE_ORDER = ("eMBB", "URLLC", "mMTC")

ACTION_WEIGHTS = np.array(
    [
        [0, 0, 0],
        [0, 0, 100],
        [0, 20, 80],
        [0, 40, 60],
        [0, 60, 40],
        [0, 80, 20],
        [0, 100, 0],
        [20, 80, 0],
        [40, 60, 0],
        [60, 40, 0],
        [80, 20, 0],
        [100, 0, 0],
        [80, 0, 20],
        [60, 0, 40],
        [40, 0, 60],
        [20, 0, 80],
        [60, 20, 20],
        [20, 60, 20],
        [20, 20, 60],
        [40, 40, 20],
        [40, 20, 40],
        [20, 40, 40],
    ],
    dtype=np.int32,
)

ACTION_SPACE_SIZE = len(ACTION_WEIGHTS)


def get_action_weights(action):
    if action < 0 or action >= ACTION_SPACE_SIZE:
        raise ValueError(
            f"Action must be in [0, {ACTION_SPACE_SIZE - 1}]"
        )

    return ACTION_WEIGHTS[action].copy()
