from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
from typing import Callable, Optional


@dataclass(frozen=True)
class MapCorrectionCompletion:
    """One completed background candidate build, successful or failed."""

    metadata: object
    value: object
    exception: Optional[Exception]
    elapsed_seconds: float


class MapCorrectionWorker:
    """Single-flight worker for immutable corrected-map candidates.

    The worker never commits node state.  The ROS callback thread takes a
    completion, validates its captured revisions, and performs the atomic
    switch.  Restricting execution to one job avoids competing full-map
    rebuilds and bounds temporary map memory.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="t3-map-correction",
        )
        self._future: Optional[Future] = None
        self._closed = False

    @property
    def active(self) -> bool:
        with self._lock:
            return self._future is not None

    @property
    def ready(self) -> bool:
        """Whether an owned job has finished and can be taken now."""
        with self._lock:
            future = self._future
        return future is not None and future.done()

    def submit(
        self,
        *,
        metadata: object,
        build: Callable[[], object],
    ) -> bool:
        """Submit when idle; return False if another job is still owned."""
        with self._lock:
            if self._closed:
                raise RuntimeError("MapCorrectionWorker is closed")
            if self._future is not None:
                return False
            self._future = self._executor.submit(
                self._execute,
                metadata,
                build,
            )
            return True

    @staticmethod
    def _execute(
        metadata: object,
        build: Callable[[], object],
    ) -> MapCorrectionCompletion:
        start = time.perf_counter()
        try:
            value = build()
        except Exception as exception:
            return MapCorrectionCompletion(
                metadata=metadata,
                value=None,
                exception=exception,
                elapsed_seconds=time.perf_counter() - start,
            )
        return MapCorrectionCompletion(
            metadata=metadata,
            value=value,
            exception=None,
            elapsed_seconds=time.perf_counter() - start,
        )

    def take_completed(
        self,
        *,
        wait: bool = False,
    ) -> Optional[MapCorrectionCompletion]:
        """Take one completion exactly once, optionally waiting for it."""
        with self._lock:
            future = self._future
        if future is None or (not wait and not future.done()):
            return None

        completion = future.result()
        with self._lock:
            if self._future is not future:
                raise RuntimeError("Map correction future ownership changed")
            self._future = None
        return completion

    def shutdown(self) -> None:
        """Wait for an active immutable build, then release its thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # ``cancel_futures`` is unavailable on the project's Python 3.8
        # deployment baseline.  There is at most one running job and it must
        # finish so its archive reads cannot race resource teardown.
        self._executor.shutdown(wait=True)
