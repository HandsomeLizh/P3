import unittest
from history_groups import BatchGroups, INPUTS
from motion_plan import describe, route


class WorkflowTests(unittest.TestCase):
    def test_durations(self):
        self.assertAlmostEqual(describe("short")["duration_seconds"], 210)
        self.assertAlmostEqual(describe("long")["duration_seconds"], 1800)
        for profile in ("short", "long"):
            curves = [leg.curvature for leg in route(profile)]
            self.assertIn(0, curves)
            self.assertTrue(any(c > 0 for c in curves))
            self.assertTrue(any(c < 0 for c in curves))

    def test_transport_delays_are_one_batch(self):
        group = BatchGroups()
        self.assertIsNone(group.add(INPUTS[0], 100, b"left"))
        self.assertIsNone(group.add(INPUTS[1], 200_000_100, b"right"))
        batch = group.add(INPUTS[2], 600_000_100, b"lidar")
        self.assertEqual(len(batch), 3)

    def test_incomplete_batches_not_mixed(self):
        group = BatchGroups()
        group.add(INPUTS[0], 0, b"old")
        group.add(INPUTS[0], 3_000_000_000, b"new")
        group.add(INPUTS[1], 3_100_000_000, b"right")
        batch = group.add(INPUTS[2], 3_500_000_000, b"lidar")
        self.assertEqual(batch[INPUTS[0]][1], b"new")
        self.assertEqual(group.dropped, 1)


if __name__ == "__main__":
    unittest.main()
