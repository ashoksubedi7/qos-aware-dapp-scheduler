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

    def record_backlog(
        self,
        value,
        cap,
    ):
        self.backlog.record(
            raw_value=float(value),
            lower_bound=0.0,
            upper_bound=float(cap),
        )

    def record_sinr(
        self,
        value,
        minimum,
        maximum,
    ):
        self.sinr.record(
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
        return {
            "backlog": asdict(self.backlog)
            | {
                "clipped_total":
                    self.backlog.clipped_total,
                "clip_rate":
                    self.backlog.clip_rate,
            },
            "sinr": asdict(self.sinr)
            | {
                "clipped_total":
                    self.sinr.clipped_total,
                "clip_rate":
                    self.sinr.clip_rate,
            },
            "urgency": asdict(self.urgency)
            | {
                "clipped_total":
                    self.urgency.clipped_total,
                "clip_rate":
                    self.urgency.clip_rate,
            },
        }
