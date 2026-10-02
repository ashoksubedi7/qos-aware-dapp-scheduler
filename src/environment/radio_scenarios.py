import hashlib
import random
from dataclasses import dataclass

VALID_RADIO_SCENARIOS = (
    "STATIC_POOR",
    "STATIC_MODERATE",
    "STATIC_GOOD",
    "STEP_DEGRADATION",
    "STEP_RECOVERY",
    "FLUCTUATING",
)

STATIC_POOR_DB = 6.0
STATIC_MODERATE_DB = 18.5
STATIC_GOOD_DB = 27.5

SINR_FLOOR_DB = 0.0
SINR_CEILING_DB = 35.0


def _stable_seed(
    experiment_seed,
    slice_name,
    ue_id,
):
    material = (
        f"{int(experiment_seed)}|"
        f"{slice_name}|"
        f"{ue_id}"
    ).encode("utf-8")

    digest = hashlib.sha256(material).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def clip_sinr(value):
    return min(
        max(
            float(value),
            SINR_FLOOR_DB,
        ),
        SINR_CEILING_DB,
    )


@dataclass
class RadioScenario:
    scenario: str
    experiment_seed: int
    slice_name: str
    ue_id: str
    simulation_time_ms: float
    update_interval_ms: float

    def __post_init__(self):
        if self.scenario not in VALID_RADIO_SCENARIOS:
            raise ValueError(f"unsupported radio scenario: {self.scenario}")

        if self.simulation_time_ms <= 0:
            raise ValueError("simulation_time_ms must be positive")

        if self.update_interval_ms <= 0:
            raise ValueError("update_interval_ms must be positive")

        self._rng = random.Random(
            _stable_seed(
                self.experiment_seed,
                self.slice_name,
                self.ue_id,
            )
        )

        self._fluctuating_cache = {0: STATIC_MODERATE_DB}

    def _fluctuating_value_at_index(
        self,
        index,
    ):
        if index < 0:
            index = 0

        if index in self._fluctuating_cache:
            return self._fluctuating_cache[index]

        last_index = max(self._fluctuating_cache.keys())
        value = self._fluctuating_cache[last_index]

        for i in range(
            last_index + 1,
            index + 1,
        ):
            innovation = self._rng.gauss(
                0.0,
                1.25,
            )

            value = (
                0.90 * value
                + 0.10 * STATIC_MODERATE_DB
                + innovation
            )

            value = clip_sinr(value)
            self._fluctuating_cache[i] = value

        return self._fluctuating_cache[index]

    def initial_sinr(self):
        if self.scenario in (
            "STATIC_GOOD",
            "STEP_DEGRADATION",
        ):
            return STATIC_GOOD_DB

        if self.scenario == "STEP_RECOVERY":
            return STATIC_POOR_DB

        if self.scenario == "STATIC_POOR":
            return STATIC_POOR_DB

        return STATIC_MODERATE_DB

    def sinr_at(
        self,
        now_ms,
    ):
        now_ms = float(now_ms)

        if self.scenario == "STATIC_POOR":
            return STATIC_POOR_DB

        if self.scenario == "STATIC_MODERATE":
            return STATIC_MODERATE_DB

        if self.scenario == "STATIC_GOOD":
            return STATIC_GOOD_DB

        midpoint = self.simulation_time_ms / 2.0

        if self.scenario == "STEP_DEGRADATION":
            if now_ms < midpoint:
                return STATIC_GOOD_DB
            return STATIC_POOR_DB

        if self.scenario == "STEP_RECOVERY":
            if now_ms < midpoint:
                return STATIC_POOR_DB
            return STATIC_GOOD_DB

        if self.scenario == "FLUCTUATING":
            index = int(
                float(now_ms) // float(self.update_interval_ms)
            )

            return self._fluctuating_value_at_index(index)

        raise RuntimeError("unreachable radio scenario")
