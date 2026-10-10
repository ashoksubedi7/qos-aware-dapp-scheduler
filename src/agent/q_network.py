from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input


def build_q_network(
    input_dim,
    hidden_units,
    action_dim,
):
    """
    Build an AssuredQoS Q-network from the
    authoritative model definition.
    """

    input_dim = int(input_dim)
    action_dim = int(action_dim)

    hidden_units = tuple(
        int(value)
        for value in hidden_units
    )

    if input_dim <= 0:
        raise ValueError(
            "input_dim must be greater than zero"
        )

    if action_dim <= 0:
        raise ValueError(
            "action_dim must be greater than zero"
        )

    if len(hidden_units) != 2:
        raise ValueError(
            "AssuredQoS Q-network requires "
            "exactly two hidden layers"
        )

    if any(
        value <= 0
        for value in hidden_units
    ):
        raise ValueError(
            "hidden units must be positive"
        )

    return Sequential(
        [
            Input(
                shape=(input_dim,)
            ),
            Dense(
                hidden_units[0],
                activation="relu",
            ),
            Dense(
                hidden_units[1],
                activation="relu",
            ),
            Dense(
                action_dim,
                activation="linear",
            ),
        ]
    )
