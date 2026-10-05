import numpy as np

from InterSliceSch import InterSliceScheduler

from assured_interval import (
    snapshot_radio_counters,
    snapshot_slice_counters,
    snapshot_urllc_deadline,
)

from assured_metrics import (
    build_m1_state,
    build_m2_state,
    slice_backlog,
    slice_mean_sinr,
    slice_max_hol_delay,
)

from agent.dqn_agent import DQNAgent
from agent.model_definitions import get_model_definition

from common.action_space import (
    ACTION_SPACE_SIZE,
    SLICE_ORDER,
    get_action_weights,
)

from common.prb_allocation import (
    percentage_action_to_numerology_prbs,
    reference_prbs_used,
)

from common.reproducibility import (
    set_global_seed,
)

from config.experiment_config import (
    ExperimentConfig,
)

from metrics.normalization_diagnostics import (
    NormalizationDiagnostics,
)

from metrics.service_metrics import (
    starvation_penalty,
)

from metrics.transition_metrics import (
    derive_transition_metrics,
)

from reward.assured_reward import (
    compute_assured_reward,
)


class AssuredScheduler(InterSliceScheduler):
    """
    Inter-slice scheduler used by AssuredQoS M1, M2 and M3.

    M1:
        backlog + slice SINR

    M2:
        backlog + slice SINR + URLLC deadline urgency

    M3:
        same state/action/model structure as M2 after
        counterexample-guided policy repair.
    """

    VALID_VARIANTS = ("M1", "M2", "M3")

    def __init__(
        self,
        ba,
        fr,
        dm,
        tdd,
        gr,
        config=None,
    ):
        super().__init__(
            ba,
            fr,
            dm,
            tdd,
            gr,
        )

        if config is None:
            config = ExperimentConfig(
                control_interval_ms=gr,
            )

        if not isinstance(
            config,
            ExperimentConfig,
        ):
            raise TypeError(
                "config must be an ExperimentConfig"
            )

        self.config = config

        set_global_seed(
            self.config.seed
        )

        self.model_variant = (
            self.config.model_variant
        )

        if (
            self.model_variant
            not in self.VALID_VARIANTS
        ):
            raise ValueError(
                "unsupported AssuredQoS model "
                f"variant: {self.model_variant}"
            )

        self.model_definition = (
            get_model_definition(
                self.model_variant
            )
        )

        self.granularity = float(
            self.config.control_interval_ms
        )

        self.urllc_deadline_ms = float(
            self.config.urllc_scheduling_deadline_ms
        )

        self.starvation_threshold_ms = float(
            self.config.starvation_threshold_ms
        )

        self.action_space_size = (
            ACTION_SPACE_SIZE
        )

        model_action_dim = int(
            self.model_definition[
                "output_dim"
            ]
        )

        if (
            model_action_dim
            != self.action_space_size
        ):
            raise ValueError(
                "model output dimension does not "
                "match centralized action space: "
                f"{model_action_dim} != "
                f"{self.action_space_size}"
            )

        self.agent = DQNAgent(
            input_dim=self.model_definition[
                "input_dim"
            ],
            action_dim=self.action_space_size,
            seed=self.config.seed,
        )

        if self.config.training_mode:
            self.agent.set_evaluation_mode(
                False
            )

        else:
            checkpoint_path = getattr(
                self.config,
                "checkpoint_path",
                None,
            )

            if not checkpoint_path:
                raise ValueError(
                    "evaluation mode requires "
                    "checkpoint_path"
                )

            self.loaded_checkpoint_metadata = (
                self.agent.load_checkpoint(
                    checkpoint_path
                )
            )

            # Must happen AFTER checkpoint loading because
            # checkpoint metadata may restore training epsilon.
            self.agent.set_evaluation_mode(
                True
            )

        if self.config.training_mode:
            self.loaded_checkpoint_metadata = (
                None
            )

        self.normalization_diagnostics = (
            NormalizationDiagnostics()
        )

        self.embb_starvation_ms = 0.0
        self.mmtc_starvation_ms = 0.0

        self.action = 0
        self.reward = 0.0

        self.last_requested_weights = None
        self.last_slice_prbs = None
        self.last_reference_prbs = None

    def build_state(self, now):
        slices = self._canonical_slices()

        common = {
            "backlog_cap":
                self.config.backlog_cap,
            "sinr_min_db":
                self.config.sinr_min_db,
            "sinr_max_db":
                self.config.sinr_max_db,
        }

        if self.model_variant == "M1":
            return build_m1_state(
                slices,
                **common,
            )

        return build_m2_state(
            slices,
            now,
            self.urllc_deadline_ms,
            **common,
        )

    def get_action_weights(self, action):
        return get_action_weights(action)

    def record_normalization_diagnostics(
        self,
        now,
    ):
        slices = self._canonical_slices()

        for name in (
            "eMBB",
            "URLLC",
            "mMTC",
        ):
            slice_obj = slices[name]

            backlog = slice_backlog(
                slice_obj
            )

            sinr = slice_mean_sinr(
                slice_obj
            )

            self.normalization_diagnostics.record_backlog(
                backlog,
                self.config.backlog_cap,
            )

            self.normalization_diagnostics.record_sinr(
                sinr,
                self.config.sinr_min_db,
                self.config.sinr_max_db,
            )

        urllc_hol = slice_max_hol_delay(
            slices["URLLC"],
            now,
        )

        raw_urgency = (
            float(urllc_hol)
            / float(
                self.config.urllc_scheduling_deadline_ms
            )
        )

        self.normalization_diagnostics.record_urgency(
            raw_urgency
        )

    def snapshot_interval_state(self):
        slices = self._canonical_slices()
        return {
            "eMBB": snapshot_slice_counters(
                slices["eMBB"]
            ),
            "URLLC": snapshot_slice_counters(
                slices["URLLC"]
            ),
            "mMTC": snapshot_slice_counters(
                slices["mMTC"]
            ),
            "radio": snapshot_radio_counters(
                slices
            ),
            "deadline": snapshot_urllc_deadline(
                slices["URLLC"]
            ),
        }

    def derive_outcomes(
        self,
        before,
        after,
        embb_target_mbps,
    ):
        return derive_transition_metrics(
            embb_before=before["eMBB"],
            embb_after=after["eMBB"],
            urllc_before=before["URLLC"],
            urllc_after=after["URLLC"],
            mmtc_before=before["mMTC"],
            mmtc_after=after["mMTC"],
            radio_before=before["radio"],
            radio_after=after["radio"],
            deadline_before=before["deadline"],
            deadline_after=after["deadline"],
            interval_ms=self.granularity,
            embb_target_mbps=embb_target_mbps,
        )

    def update_starvation(
        self,
        before,
        after,
    ):
        embb_bytes = max(
            0,
            after["eMBB"].delivered_bytes
            - before["eMBB"].delivered_bytes,
        )

        mmtc_bytes = max(
            0,
            after["mMTC"].delivered_bytes
            - before["mMTC"].delivered_bytes,
        )

        self.embb_starvation_ms, embb_penalty = (
            starvation_penalty(
                backlog_before=before["eMBB"].backlog,
                delivered_bytes=embb_bytes,
                previous_starvation_ms=self.embb_starvation_ms,
                interval_ms=self.granularity,
                threshold_ms=self.starvation_threshold_ms,
            )
        )

        self.mmtc_starvation_ms, mmtc_penalty = (
            starvation_penalty(
                backlog_before=before["mMTC"].backlog,
                delivered_bytes=mmtc_bytes,
                previous_starvation_ms=self.mmtc_starvation_ms,
                interval_ms=self.granularity,
                threshold_ms=self.starvation_threshold_ms,
            )
        )

        return max(
            embb_penalty,
            mmtc_penalty,
        )

    def compute_reward(
        self,
        outcomes,
        starvation_value,
    ):
        weights = {
            "urllc":
                self.config.reward_urllc_service,

            "embb":
                self.config.reward_embb_service,

            "mmtc":
                self.config.reward_mmtc_service,

            "utilization":
                self.config.reward_utilization,

            "deadline":
                self.config.reward_deadline,

            "starvation":
                self.config.reward_starvation,
        }

        return compute_assured_reward(
            urllc_service=(
                outcomes.urllc_service
            ),
            embb_service=(
                outcomes.embb_service
            ),
            mmtc_service=(
                outcomes.mmtc_service
            ),
            utilization=(
                outcomes.utilization
            ),
            deadline_miss_ratio=(
                outcomes.deadline_miss_ratio
            ),
            starvation_penalty=(
                starvation_value
            ),
            weights=weights,
        )

    def _find_slice_key(self, service):
        matches = [
            key
            for key in self.slices
            if service in key
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {service} slice, "
                f"found {matches}"
            )

        return matches[0]

    def _canonical_slices(self):
        return {
            "eMBB": self.slices[
                self._find_slice_key("eMBB")
            ],
            "URLLC": self.slices[
                self._find_slice_key("URLLC")
            ],
            "mMTC": self.slices[
                self._find_slice_key("mMTC")
            ],
        }

    def apply_action(self, action):
        """
        Convert a learned action into feasible per-slice PRB allocations.

        The DQN chooses a percentage split in common reference-resource
        units. This method performs only deterministic feasibility
        conversion; it does not add QoS priorities or safety overrides.
        """
        slices = self._canonical_slices()

        weights = tuple(
            int(value)
            for value in self.get_action_weights(action)
        )

        factors = tuple(
            int(slices[name].numRefFactor)
            for name in SLICE_ORDER
        )

        allocation = percentage_action_to_numerology_prbs(
            total_reference_prbs=self.PRBs,
            weights=weights,
            num_ref_factors=factors,
        )

        for name, prbs in zip(
            SLICE_ORDER,
            allocation,
        ):
            slices[name].updateConfig(
                int(prbs)
            )

        reference_allocation = tuple(
            int(allocation[index])
            * int(factors[index])
            for index in range(len(SLICE_ORDER))
        )

        self.last_requested_weights = weights
        self.last_slice_prbs = allocation
        self.last_reference_prbs = reference_allocation

        return {
            "action": int(action),
            "weights": weights,
            "slice_prbs": allocation,
            "reference_prbs": reference_allocation,
            "reference_prbs_used": reference_prbs_used(
                allocation,
                factors,
            ),
            "reference_prbs_available": int(self.PRBs),
        }

    def _embb_target_mbps(self):
        slices = self._canonical_slices()
        embb = slices["eMBB"]

        num_ues = len(
            embb.schedulerDL.ues
        )

        return (
            float(embb.reqThroughputDL)
            * float(num_ues)
        )

    def select_policy_action(self, state):
        return self.agent.select_action(
            state,
            explore=self.config.training_mode,
        )

    def resAlloc(self, env):
        """
        AssuredQoS inter-slice RL control loop.
        """

        while True:
            if len(self.slices) < 3:
                yield env.timeout(
                    self.granularity
                )
                continue
                
            self.record_normalization_diagnostics(
                env.now
            )
            state = np.asarray(
                self.build_state(env.now),
                dtype=np.float32,
            )

            before = (
                self.snapshot_interval_state()
            )

            action = self.select_policy_action(
                state,
            )

            self.action = int(action)

            self.apply_action(
                self.action
            )

            yield env.timeout(
                self.granularity
            )

            after = (
                self.snapshot_interval_state()
            )

            outcomes = self.derive_outcomes(
                before,
                after,
                embb_target_mbps=(
                    self._embb_target_mbps()
                ),
            )

            starvation_value = (
                self.update_starvation(
                    before,
                    after,
                )
            )

            reward_components = (
                self.compute_reward(
                    outcomes,
                    starvation_value,
                )
            )

            self.reward = float(
                reward_components.total
            )

            next_state = np.asarray(
                self.build_state(env.now),
                dtype=np.float32,
            )

            if self.config.training_mode:
                self.agent.remember(
                    state,
                    self.action,
                    self.reward,
                    next_state,
                    False,
                )

                self.agent.train_step()
                self.agent.decay_epsilon()
