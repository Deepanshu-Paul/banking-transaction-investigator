import time


class Deadline:
    def __init__(self, timeout_seconds: float):
        self.expires_at = time.monotonic() + timeout_seconds

    def remaining_seconds(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())

    def expired(self) -> bool:
        return self.remaining_seconds() <= 0