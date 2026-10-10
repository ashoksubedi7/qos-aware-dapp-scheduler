import json
import random
from collections import deque
from pathlib import Path

import numpy as np
from tensorflow.keras.optimizers import Adam

from agent.q_network import build_q_network
from config.dqn_hyperparameters import (
    DQNHyperparameters,
)


class DQNAgent:
    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_units,
        hyperparameters,
        seed=None,
        self_evaluation_mode=False,
    ):
        if input_dim <= 0:
            raise ValueError(
                "input_dim must be greater than zero"
            )

        if action_dim <= 0:
            raise ValueError(
                "action_dim must be greater than zero"
            )

        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)

        if not isinstance(
            hyperparameters,
            DQNHyperparameters,
        ):
            raise TypeError(
                "hyperparameters must be a "
                "DQNHyperparameters instance"
            )

        self.hidden_units = tuple(
            int(value)
            for value in hidden_units
        )

        if len(self.hidden_units) != 2:
            raise ValueError(
                "AssuredQoS requires exactly "
                "two hidden layers"
            )

        if any(
            value <= 0
            for value in self.hidden_units
        ):
            raise ValueError(
                "hidden units must be positive"
            )

        self.hyperparameters = hyperparameters

        self.gamma = float(
            hyperparameters.gamma
        )

        self.batch_size = int(
            hyperparameters.batch_size
        )

        self.min_replay_size = int(
            hyperparameters.min_replay_size
        )

        self.target_update_interval = int(
            hyperparameters.target_update_interval
        )

        self.epsilon = float(
            hyperparameters.epsilon_start
        )

        self.epsilon_min = float(
            hyperparameters.epsilon_min
        )

        self.epsilon_decay = float(
            hyperparameters.epsilon_decay
        )

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(
            seed
        )

        self.replay_buffer = deque(
            maxlen=int(
                hyperparameters.replay_capacity
            )
        )

        self.online_model = build_q_network(
            input_dim=self.input_dim,
            hidden_units=self.hidden_units,
            action_dim=self.action_dim,
        )

        self.target_model = build_q_network(
            input_dim=self.input_dim,
            hidden_units=self.hidden_units,
            action_dim=self.action_dim,
        )

        self.online_model.compile(
            optimizer=Adam(
                learning_rate=(
                    hyperparameters.learning_rate
                )
            ),
            loss="mse",
        )

        self.target_model.compile(
            optimizer=Adam(
                learning_rate=(
                    hyperparameters.learning_rate
                )
            ),
            loss="mse",
        )

        self.training_steps = 0

        self.evaluation_mode = bool(
            self_evaluation_mode
        )

        self.update_target_network()

        if self.evaluation_mode:
            self.epsilon = 0.0

    @property
    def model(self):
        """
        Alias for compatibility with
        external callers and tests.
        """
        return self.online_model

    def _prepare_state(
        self,
        state,
    ):
        state = np.asarray(
            state,
            dtype=np.float32,
        )

        if state.shape != (
            self.input_dim,
        ):
            raise ValueError(
                "expected state shape "
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

        if self.evaluation_mode:
            explore = False

        if (
            explore
            and self.rng.random()
            < self.epsilon
        ):
            return self.rng.randrange(
                self.action_dim
            )

        q_values = (
            self.online_model.predict(
                state.reshape(
                    1,
                    -1,
                ),
                verbose=0,
            )[0]
        )

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
        if self.evaluation_mode:
            return

        state = self._prepare_state(
            state
        )

        next_state = self._prepare_state(
            next_state
        )

        action = int(action)

        if not (
            0
            <= action
            < self.action_dim
        ):
            raise ValueError(
                "action outside action space"
            )

        self.replay_buffer.append(
            (
                state.copy(),
                action,
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
        if self.evaluation_mode:
            return None

        if not self.can_train():
            return None

        batch = self.rng.sample(
            self.replay_buffer,
            self.batch_size,
        )

        states = np.asarray(
            [
                item[0]
                for item in batch
            ],
            dtype=np.float32,
        )

        actions = np.asarray(
            [
                item[1]
                for item in batch
            ],
            dtype=np.int64,
        )

        rewards = np.asarray(
            [
                item[2]
                for item in batch
            ],
            dtype=np.float32,
        )

        next_states = np.asarray(
            [
                item[3]
                for item in batch
            ],
            dtype=np.float32,
        )

        dones = np.asarray(
            [
                item[4]
                for item in batch
            ],
            dtype=np.float32,
        )

        current_q = (
            self.online_model.predict(
                states,
                verbose=0,
            )
        )

        next_q = (
            self.target_model.predict(
                next_states,
                verbose=0,
            )
        )

        targets = current_q.copy()

        max_next_q = np.max(
            next_q,
            axis=1,
        )

        target_values = (
            rewards
            + (
                1.0
                - dones
            )
            * self.gamma
            * max_next_q
        )

        targets[
            np.arange(
                self.batch_size
            ),
            actions,
        ] = target_values

        history = (
            self.online_model.fit(
                states,
                targets,
                batch_size=(
                    self.batch_size
                ),
                epochs=1,
                verbose=0,
            )
        )

        self.training_steps += 1

        if (
            self.training_steps
            % self.target_update_interval
            == 0
        ):
            self.update_target_network()

        return float(
            history.history[
                "loss"
            ][0]
        )

    def update_target_network(
        self,
    ):
        if self.evaluation_mode:
            return

        self.target_model.set_weights(
            self.online_model.get_weights()
        )

    def save_checkpoint(
        self,
        directory,
        metadata=None,
    ):
        directory = Path(
            directory
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        online_path = (
            directory
            / "online.weights.h5"
        )

        target_path = (
            directory
            / "target.weights.h5"
        )

        metadata_path = (
            directory
            / "metadata.json"
        )

        self.online_model.save_weights(
            online_path
        )

        self.target_model.save_weights(
            target_path
        )

        checkpoint_metadata = {
            "input_dim": int(
                self.input_dim
            ),
            "action_dim": int(
                self.action_dim
            ),
            "epsilon": float(
                self.epsilon
            ),
            "training_steps": int(
                self.training_steps
            ),
        }

        if metadata is not None:
            checkpoint_metadata.update(
                metadata
            )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                checkpoint_metadata,
                handle,
                indent=2,
                sort_keys=True,
            )

        return {
            "online_weights": str(
                online_path
            ),
            "target_weights": str(
                target_path
            ),
            "metadata": str(
                metadata_path
            ),
        }

    def load_checkpoint(
        self,
        directory,
    ):
        directory = Path(
            directory
        )

        online_path = (
            directory
            / "online.weights.h5"
        )

        target_path = (
            directory
            / "target.weights.h5"
        )

        metadata_path = (
            directory
            / "metadata.json"
        )

        if not online_path.exists():
            raise FileNotFoundError(
                "Checkpoint not found at "
                f"{online_path}"
            )

        metadata = {}

        if metadata_path.exists():
            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                metadata = json.load(
                    handle
                )

            if (
                "input_dim"
                in metadata
                and int(
                    metadata[
                        "input_dim"
                    ]
                )
                != self.input_dim
            ):
                raise ValueError(
                    "checkpoint input dimension "
                    "does not match agent"
                )

            if (
                "action_dim"
                in metadata
                and int(
                    metadata[
                        "action_dim"
                    ]
                )
                != self.action_dim
            ):
                raise ValueError(
                    "checkpoint action dimension "
                    "does not match agent"
                )

        self.online_model.load_weights(
            online_path
        )

        if target_path.exists():
            self.target_model.load_weights(
                target_path
            )
        else:
            self.target_model.set_weights(
                self.online_model.get_weights()
            )

        if "epsilon" in metadata:
            self.epsilon = float(
                metadata["epsilon"]
            )

        if (
            "training_steps"
            in metadata
        ):
            self.training_steps = int(
                metadata[
                    "training_steps"
                ]
            )

        return metadata

    def set_evaluation_mode(
        self,
        enabled=True,
    ):
        self.evaluation_mode = bool(
            enabled
        )

        if self.evaluation_mode:
            self.epsilon = 0.0

    def decay_epsilon(
        self,
    ):
        if self.evaluation_mode:
            return 0.0

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            * self.epsilon_decay,
        )

        return self.epsilon
