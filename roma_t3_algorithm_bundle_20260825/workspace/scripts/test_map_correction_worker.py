#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import threading
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "src" / "t3_semantic_mapping"
sys.path.insert(0, str(SOURCE))

from t3_semantic_mapping.map_correction_worker import MapCorrectionWorker


class MapCorrectionWorkerTest(unittest.TestCase):
    def test_single_flight_completion_is_taken_once(self):
        worker = MapCorrectionWorker()
        release = threading.Event()
        try:
            self.assertTrue(
                worker.submit(
                    metadata="revision-3",
                    build=lambda: (release.wait(), 17)[1],
                )
            )
            self.assertFalse(
                worker.submit(metadata="revision-4", build=lambda: 18)
            )
            self.assertFalse(worker.ready)
            self.assertIsNone(worker.take_completed())
            release.set()
            completion = worker.take_completed(wait=True)
            self.assertEqual(completion.metadata, "revision-3")
            self.assertEqual(completion.value, 17)
            self.assertIsNone(completion.exception)
            self.assertFalse(worker.ready)
            self.assertIsNone(worker.take_completed())
        finally:
            release.set()
            worker.shutdown()

    def test_failure_is_returned_without_killing_next_job(self):
        worker = MapCorrectionWorker()
        try:
            def fail():
                raise RuntimeError("synthetic rebuild failure")

            self.assertTrue(worker.submit(metadata=1, build=fail))
            failed = worker.take_completed(wait=True)
            self.assertIsInstance(failed.exception, RuntimeError)
            self.assertTrue(worker.submit(metadata=2, build=lambda: 9))
            succeeded = worker.take_completed(wait=True)
            self.assertEqual(succeeded.value, 9)
            self.assertIsNone(succeeded.exception)
        finally:
            worker.shutdown()

    def test_shutdown_waits_and_rejects_new_work(self):
        worker = MapCorrectionWorker()
        release = threading.Event()
        self.assertTrue(
            worker.submit(metadata=None, build=lambda: release.wait())
        )
        release.set()
        worker.shutdown()
        worker.shutdown()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            worker.submit(metadata=None, build=lambda: None)


if __name__ == "__main__":
    unittest.main()
