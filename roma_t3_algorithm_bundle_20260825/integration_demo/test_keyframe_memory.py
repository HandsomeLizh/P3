import unittest
import numpy as np
from roma_vo import KeyFrame, StereoSLAMv2, make_keyframe_geometry_snapshot


class ImageRetentionTests(unittest.TestCase):
    def test_pixels_released_but_geometry_and_image_size_preserved(self):
        slam = StereoSLAMv2.__new__(StereoSLAMv2)
        k = np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1]])
        slam.keyframes = [KeyFrame(id=i, timestamp=float(i), pose=np.eye(4),
                                   image=np.zeros((6, 9, 3), dtype=np.uint8),
                                   keypoints=[[1., 2.]], camera_points=[[0., 0., 1.]],
                                   descriptors=np.ones((1, 32), dtype=np.uint8), K=k)
                          for i in range(5)]
        before = make_keyframe_geometry_snapshot(slam.keyframes[0], k)
        self.assertEqual(slam.trim_keyframe_images(2), 3)
        self.assertIsNone(slam.keyframes[0].image)
        self.assertEqual(sum(item.image is not None for item in slam.keyframes), 2)
        after = make_keyframe_geometry_snapshot(slam.keyframes[0], k)
        self.assertEqual(after.image_size, (6, 9))
        np.testing.assert_array_equal(before.camera_points, after.camera_points)
        np.testing.assert_array_equal(before.descriptors, after.descriptors)
        self.assertEqual(slam.trim_keyframe_images(2), 0)

    def test_default_keeps_images(self):
        slam = StereoSLAMv2.__new__(StereoSLAMv2)
        slam.keyframes = [KeyFrame(id=0, timestamp=0, pose=np.eye(4), image=np.zeros((2, 2, 3)))]
        self.assertEqual(slam.trim_keyframe_images(0), 0)
        self.assertIsNotNone(slam.keyframes[0].image)

    def test_long_sequence_has_bounded_pixel_cache(self):
        slam = StereoSLAMv2.__new__(StereoSLAMv2)
        slam.keyframes = []
        for index in range(500):
            slam.keyframes.append(KeyFrame(
                id=index, timestamp=float(index), pose=np.eye(4),
                image=np.zeros((6, 9, 3), dtype=np.uint8)))
            slam.trim_keyframe_images(8)
            self.assertLessEqual(sum(k.image is not None for k in slam.keyframes), 8)
        self.assertEqual(len(slam.keyframes), 500)
        self.assertEqual(slam.keyframes[0].image_size_hint, (6, 9))

    def test_existing_positional_constructor_is_unchanged(self):
        keypoints = [[1., 2.]]
        keyframe = KeyFrame(0, 0., np.eye(4), None, keypoints)
        self.assertIs(keyframe.keypoints, keypoints)


if __name__ == "__main__":
    unittest.main()
