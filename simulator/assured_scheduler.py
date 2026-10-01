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
)

from agent.model_definitions import get_model_definition
from common.action_space import (
    ACTION_SPACE_SIZE,
    get_action_weights,
)
from metrics.service_metrics import starvation_penalty
from metrics.transition_metrics import derive_transition_metrics
from reward.assured_reward import compute_assured_reward


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
        model_variant="M1",
        urllc_deadline_ms=1.0,
        starvation_threshold_ms=10.0,
    ):
        super().__init__(
            ba,
            fr,
            dm,
            tdd,
            gr,
        )

        if model_variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unknown model variant: {model_variant}"
            )

        if urllc_deadline_ms <= 0:
            raise ValueError(
                "URLLC deadline must be greater than zero"
            )

        if starvation_threshold_ms <= 0:
            raise ValueError(
                "Starvation threshold must be greater than zero"
            )

        self.model_variant = model_variant
        self.model_definition = get_model_definition(
            model_variant
        )

        self.urllc_deadline_ms = float(
            urllc_deadline_ms
        )

        self.starvation_threshold_ms = float(
            starvation_threshold_ms
        )

        self.action_space_size = ACTION_SPACE_SIZE

        self.embb_starvation_ms = 0.0
        self.mmtc_starvation_ms = 0.0

        self.action = 0
        self.reward = 0.0

    def build_state(self, now):
        if self.model_variant == "M1":
            return build_m1_state(
                self.slices
            )

        return build_m2_state(
            self.slices,
            now,
            self.urllc_deadline_ms,
        )

    def get_action_weights(self, action):
        return get_action_weights(action)

    def snapshot_interval_state(self):
        return {
            "eMBB": snapshot_slice_counters(
                self.slices["eMBB"]
            ),
            "URLLC": snapshot_slice_counters(
                self.slices["URLLC"]
            ),
            "mMTC": snapshot_slice_counters(
                self.slices["mMTC"]
            ),
            "radio": snapshot_radio_counters(
                self.slices
            ),
            "deadline": snapshot_urllc_deadline(
                self.slices["URLLC"]
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
        return compute_assured_reward(
            urllc_service=outcomes.urllc_service,
            embb_service=outcomes.embb_service,
            mmtc_service=outcomes.mmtc_service,
            utilization=outcomes.utilization,
            deadline_miss_ratio=outcomes.deadline_miss_ratio,
            starvation_penalty=starvation_value,
        )
