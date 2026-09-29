from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input


ACTION_SPACE_SIZE = 22


def build_q_network(input_dim, hidden_units=(64, 64)):
    """Build the Q-network used by the AssuredQoS M1/M2/M3 policies."""

    if input_dim <= 0:
        raise ValueError("input_dim must be greater than zero")

    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(hidden_units[0], activation="relu"),
            Dense(hidden_units[1], activation="relu"),
            Dense(ACTION_SPACE_SIZE, activation="linear"),
        ]
    )

    return model
