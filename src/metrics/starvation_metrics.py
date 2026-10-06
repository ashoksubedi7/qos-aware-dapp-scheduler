from dataclasses import dataclass


@dataclass
class StarvationTracker:
    threshold_ms: float

    current_ms: float = 0.0
    total_ms: float = 0.0
    max_ms: float = 0.0
    event_count: int = 0

    _event_counted: bool = False

    def __post_init__(self):
        if self.threshold_ms <= 0:
            raise ValueError(
                "threshold_ms must be "
                "greater than zero"
            )

    def update(
        self,
        backlog_before,
        delivered_bytes,
        interval_ms,
    ):
        if interval_ms <= 0:
            raise ValueError(
                "interval_ms must be "
                "greater than zero"
            )

        starving = (
            int(backlog_before) > 0
            and int(delivered_bytes) <= 0
        )

        if starving:
            self.current_ms += float(
                interval_ms
            )

            self.total_ms += float(
                interval_ms
            )

            self.max_ms = max(
                self.max_ms,
                self.current_ms,
            )

            if (
                self.current_ms
                >= self.threshold_ms
                and not self._event_counted
            ):
                self.event_count += 1
                self._event_counted = True

        else:
            self.current_ms = 0.0
            self._event_counted = False

        return starving

from dataclasses import dataclass


@dataclass
class StarvationTracker:
    threshold_ms: float

    current_ms: float = 0.0
    total_ms: float = 0.0
    max_ms: float = 0.0
    event_count: int = 0

    _event_counted: bool = False

    def __post_init__(self):
        if self.threshold_ms <= 0:
            raise ValueError(
                "threshold_ms must be "
                "greater than zero"
            )

    def update(
        self,
        backlog_before,
        delivered_bytes,
        interval_ms,
    ):
        if interval_ms <= 0:
            raise ValueError(
                "interval_ms must be "
                "greater than zero"
            )

        starving = (
            int(backlog_before) > 0
            and int(delivered_bytes) <= 0
        )

        if starving:
            self.current_ms += float(
                interval_ms
            )

            self.total_ms += float(
                interval_ms
            )

            self.max_ms = max(
                self.max_ms,
                self.current_ms,
            )

            if (
                self.current_ms
                >= self.threshold_ms
                and not self._event_counted
            ):
                self.event_count += 1
                self._event_counted = True

        else:
            self.current_ms = 0.0
            self._event_counted = False

        return starving

