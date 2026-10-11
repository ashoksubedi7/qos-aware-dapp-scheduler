from dataclasses import dataclass, asdict


@dataclass
class FeatureClipStats:
    observations: int = 0
    clipped_low: int = 0
    clipped_high: int = 0

    def record(
        self,
        raw_value,
        lower_bound,
        upper_bound,
    ):
        self.observations += 1

        if raw_value < lower_bound:
            self.clipped_low += 1

        if raw_value > upper_bound:
            self.clipped_high += 1

    @property
    def clipped_total(self):
        return (
            self.clipped_low
            + self.clipped_high
        )

    @property
    def clip_rate(self):
        if self.observations == 0:
            return 0.0

        return (
            self.clipped_total
            / self.observations
        )


class NormalizationDiagnostics:
    """
    Track how often raw AssuredQoS state variables exceed
    their configured normalization domains.
    """

    def __init__(self):
        self.backlog = FeatureClipStats()
        self.sinr = FeatureClipStats()
        self.urgency = FeatureClipStats()

        self.backlog_by_slice = {
            name: FeatureClipStats()
            for name in (
                "eMBB",
                "URLLC",
                "mMTC",
            )
        }
        self.sinr_by_slice = {
            name: FeatureClipStats()
            for name in (
                "eMBB",
                "URLLC",
                "mMTC",
            )
        }

    def record_backlog(
        self,
        value,
        cap,
        slice_name=None,
    ):
        self.backlog.record(
            raw_value=float(value),
            lower_bound=0.0,
            upper_bound=float(cap),
        )

        if slice_name is None:
            return

        if slice_name not in self.backlog_by_slice:
            raise ValueError(
                "Unknown slice for backlog diagnostics: "
                f"{slice_name}"
            )

        self.backlog_by_slice[
            slice_name
        ].record(
            raw_value=float(value),
            lower_bound=0.0,
            upper_bound=float(cap),
        )

    def record_sinr(
        self,
        value,
        minimum,
        maximum,
        slice_name=None,
    ):
        self.sinr.record(
            raw_value=float(value),
            lower_bound=float(minimum),
            upper_bound=float(maximum),
        )

        if slice_name is None:
            return

        if slice_name not in self.sinr_by_slice:
            raise ValueError(
                "Unknown slice for SINR diagnostics: "
                f"{slice_name}"
            )

        self.sinr_by_slice[
            slice_name
        ].record(
            raw_value=float(value),
            lower_bound=float(minimum),
            upper_bound=float(maximum),
        )

    def record_urgency(
        self,
        value,
    ):
        self.urgency.record(
            raw_value=float(value),
            lower_bound=0.0,
            upper_bound=1.0,
        )

    def to_dict(self):
        def serialize(stats):
            return asdict(stats) | {
                "clipped_total":
                    stats.clipped_total,
                "clip_rate":
                    stats.clip_rate,
            }

        return {
            "backlog":
                serialize(self.backlog),
            "backlog_by_slice": {
                name: serialize(stats)
                for name, stats
                in self.backlog_by_slice.items()
            },
            "sinr":
                serialize(self.sinr),
            "sinr_by_slice": {
                name: serialize(stats)
                for name, stats
                in self.sinr_by_slice.items()
            },
            "urgency":
                serialize(self.urgency),
        }
