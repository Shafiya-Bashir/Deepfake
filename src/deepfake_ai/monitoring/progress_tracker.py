"""Progress tracking and ETA estimation for epochs and dataset processing."""

import time
from typing import Optional


class ProgressTracker:
    """Calculates iteration rates, remaining time, and elapsed duration."""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        self.current_step += n

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def eta_seconds(self) -> float:
        if self.current_step == 0:
            return 0.0
        rate = self.elapsed_seconds / self.current_step
        return rate * (self.total_steps - self.current_step)
