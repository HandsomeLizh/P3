import argparse
import unittest
from live_demo import commands


class LivePlanTests(unittest.TestCase):
    def args(self, action):
        return argparse.Namespace(action=action, host='192.168.10.22', capture_port=6665, car_port=6668, capture_hz=.25)

    def test_start_is_acquisition_only_without_bag_or_motion(self):
        plan = commands(self.args('start'))
        self.assertEqual(len(plan), 2)
        self.assertIn('--capture-only', plan[1])
        self.assertIn('10', plan[0])
        self.assertNotIn('gui', str(plan))
        self.assertNotIn('replay_history', str(plan))
        self.assertNotIn('fixed_motion', str(plan))

    def test_vehicle_connection_is_separate_explicit_action(self):
        plan = commands(self.args('control'))
        self.assertNotIn('--capture-only', plan[0])
        self.assertEqual(plan[-1][-1], 'gui')
        self.assertNotIn('fixed_motion', str(plan))


if __name__ == '__main__':
    unittest.main()
