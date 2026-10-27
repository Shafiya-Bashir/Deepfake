"""Execution timing context managers."""

import time
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class Timer:
    """Simple stopwatch timer."""
    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
            self.start_time = None
        return self.elapsed


@contextmanager
def time_block(description: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f'{description} completed in {elapsed:.4f} seconds')
