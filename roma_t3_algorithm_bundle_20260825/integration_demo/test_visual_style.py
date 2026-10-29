"""Pure display math regression; never starts a GUI or a vehicle node."""
import unittest
import numpy as np
from visual_style import display_sample, expand_height_limits, height_colors, localization_status, HEIGHT_STOPS, UNKNOWN_COLOR


class VisualStyleTests(unittest.TestCase):
    def test_finished_input_is_not_reported_as_hundreds_of_seconds_stale(self):
        label, _ = localization_status(600., has_path=True, phase='completed')
        self.assertIn('已结束', label)
        self.assertNotIn('600', label)
        label, _ = localization_status(600., has_path=True)
        self.assertIn('检查定位数据', label)
        self.assertIn('600', label)
        label, _ = localization_status(1., phase='error')
        self.assertIn('异常停止', label)

    def test_continuous_gradient_endpoints_and_midpoint(self):
        colors = height_colors(np.array([-5., 0., 5.]), -5., 5.)
        np.testing.assert_array_equal(colors, HEIGHT_STOPS.astype(np.uint8))
        ramp = height_colors(np.linspace(-5, 5, 256), -5., 5.)
        self.assertGreater(len(np.unique(ramp, axis=0)), 200)

    def test_unknown_not_low_elevation(self):
        colors = height_colors(np.array([np.nan, -2., 5.]), -2., 5.)
        np.testing.assert_array_equal(colors[0], UNKNOWN_COLOR)
        self.assertFalse(np.array_equal(colors[0], colors[1]))

    def test_expanding_limits_do_not_change_per_frame_for_same_range(self):
        self.assertEqual(expand_height_limits([-3.2, 2.4]), (-4., 3.))
        self.assertEqual(expand_height_limits([-1., 1.], (-4., 3.)), (-4., 3.))
        self.assertEqual(expand_height_limits([-4.2, 3.1], (-4., 3.)), (-5., 4.))
        self.assertEqual(expand_height_limits([0.]), (0., 1.))

    def test_sampling_is_finite_bounded_and_does_not_modify_input(self):
        points = np.random.default_rng(1).normal(size=(200000, 3)) * 200
        points[0] = np.nan
        before = points.copy()
        sampled = display_sample(points)
        self.assertLessEqual(len(sampled), 50000)
        self.assertTrue(np.isfinite(sampled).all())
        np.testing.assert_array_equal(points, before)

    def test_one_point_per_voxel_and_empty_input(self):
        samples = display_sample([[0, 0, 0], [.1, .1, .1], [.6, .6, .6]])
        self.assertEqual(len(samples), 2)
        self.assertEqual(display_sample(np.empty((0, 3))).shape, (0, 3))


if __name__ == '__main__':
    unittest.main()
