import random
from collections import deque

import numpy as np
from tensorflow.keras.optimizers import Adam

from agent.q_network import build_q_network


class DQNAgent:
    def __init__(
        self,
        input_dim,
        action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        replay_capacity=100_000,
        batch_size=64,
        min_replay_size=1_000,
        target_update_interval=250,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
        seed=None,
    ):
        if input_dim <= 0:
            raise ValueError("input_dim must be greater than zero")

        if action_dim <= 0:
            raise ValueError("action_dim must be greater than zero")

        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")

        if replay_capacity <= 0:
            raise ValueError("replay_capacity must be greater than zero")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if min_replay_size < batch_size:
            raise ValueError(
                "min_replay_size must be >= batch_size"
            )

        if target_update_interval <= 0:
            raise ValueError(
                "target_update_interval must be greater than zero"
            )

        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)

        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.min_replay_size = int(min_replay_size)
        self.target_update_interval = int(
            target_update_interval
        )

        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.replay_buffer = deque(
            maxlen=int(replay_capacity)
        )

        self.online_model = build_q_network(
            input_dim=self.input_dim,
        )

        self.target_model = build_q_network(
            input_dim=self.input_dim,
        )

        optimizer = Adam(
            learning_rate=learning_rate
        )

        self.online_model.compile(
            optimizer=optimizer,
            loss="mse",
        )

        self.target_model.compile(
            optimizer=Adam(
                learning_rate=learning_rate
            ),
            loss="mse",
        )

        self.update_target_network()

        self.training_steps = 0

    def _prepare_state(self, state):
        state = np.asarray(
            state,
            dtype=np.float32,
        )

        if state.shape != (self.input_dim,):
            raise ValueError(
                f"expected state shape "
                f"({self.input_dim},), "
                f"got {state.shape}"
            )

        return state

    def select_action(
        self,
        state,
        explore=True,
    ):
        state = self._prepare_state(
            state
        )

        if (
            explore
            and self.rng.random()
            < self.epsilon
        ):
            return self.rng.randrange(
                self.action_dim
            )

        q_values = self.online_model.predict(
            state.reshape(1, -1),
            verbose=0,
        )[0]

        return int(
            np.argmax(q_values)
        )

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        state = self._prepare_state(
            state
        )

        next_state = self._prepare_state(
            next_state
        )

        if not 0 <= int(action) < self.action_dim:
            raise ValueError(
                "action outside action space"
            )

        self.replay_buffer.append(
            (
                state.copy(),
                int(action),
                float(reward),
                next_state.copy(),
                bool(done),
            )
        )

    def can_train(self):
        return (
            len(self.replay_buffer)
            >= self.min_replay_size
        )

    def train_step(self):
        if not self.can_train():
            return None

        batch = self.rng.sample(
            self.replay_buffer,
            self.batch_size,
        )

        states = np.asarray(
            [item[0] for item in batch],
            dtype=np.float32,
        )

        actions = np.asarray(
            [item[1] for item in batch],
            dtype=np.int64,
        )

        rewards = np.asarray(
            [item[2] for item in batch],
            dtype=np.float32,
        )

        next_states = np.asarray(
            [item[3] for item in batch],
            dtype=np.float32,
        )

        dones = np.asarray(
            [item[4] for item in batch],
            dtype=np.float32,
        )

        current_q = self.online_model.predict(
            states,
            verbose=0,
        )

        next_q = self.target_model.predict(
            next_states,
            verbose=0,
        )

        targets = current_q.copy()

        max_next_q = np.max(
            next_q,
            axis=1,
        )

        target_values = (
            rewards
            + (1.0 - dones)
            * self.gamma
            * max_next_q
        )

        targets[
            np.arange(self.batch_size),
            actions,
        ] = target_values

        history = self.online_model.fit(
            states,
            targets,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0,
        )

        self.training_steps += 1

        if (
            self.training_steps
            % self.target_update_interval
            == 0
        ):
            self.update_target_network()

        return float(
            history.history["loss"][0]
        )

    def update_target_network(self):
        self.target_model.set_weights(
            self.online_model.get_weights()
        )

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            * self.epsilon_decay,
        )

        return self.epsilon
