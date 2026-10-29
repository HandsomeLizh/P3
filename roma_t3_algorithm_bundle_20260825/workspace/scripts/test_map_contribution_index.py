#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "src" / "t3_semantic_mapping"
sys.path.insert(0, str(SOURCE))

from t3_semantic_mapping.map_contribution_index import MapContributionIndex


def _pose(x: float = 0.0) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[0, 3] = float(x)
    return result


class MapContributionIndexTest(unittest.TestCase):
    def test_reverse_lookup_tracks_record_replacement(self):
        index = MapContributionIndex()
        index.record(
            keyframe_id=3,
            pose_world_camera=_pose(),
            tile_keys=((0, 0), (1, 0)),
        )
        index.record(
            keyframe_id=7,
            pose_world_camera=_pose(1.0),
            tile_keys=((1, 0), (2, 0)),
        )

        self.assertEqual(index.contributors(((1, 0),)), (3, 7))
        index.record(
            keyframe_id=3,
            pose_world_camera=_pose(2.0),
            tile_keys=((-1, 0),),
        )

        self.assertEqual(index.contributors(((1, 0),)), (7,))
        self.assertEqual(index.contributors(((-1, 0),)), (3,))
        index.assert_consistent()

    def test_clone_is_independent_and_pose_reads_are_detached(self):
        index = MapContributionIndex()
        index.record(
            keyframe_id=4,
            pose_world_camera=_pose(),
            tile_keys=((0, 0),),
        )
        candidate = index.clone()
        candidate.record(
            keyframe_id=4,
            pose_world_camera=_pose(5.0),
            tile_keys=((5, 0),),
        )
        detached_pose = index.pose(4)
        assert detached_pose is not None
        detached_pose[0, 3] = 99.0

        self.assertEqual(index.tile_keys(4), frozenset({(0, 0)}))
        self.assertEqual(candidate.tile_keys(4), frozenset({(5, 0)}))
        self.assertEqual(float(index.pose(4)[0, 3]), 0.0)
        self.assertEqual(float(candidate.pose(4)[0, 3]), 5.0)

    def test_exact_pose_change_and_complete_keyframe_set(self):
        index = MapContributionIndex()
        index.record_many(
            poses_world_camera={2: _pose(), 5: _pose(1.0)},
            tile_keys_by_keyframe={2: ((0, 0),), 5: ((1, 0),)},
        )

        self.assertTrue(index.is_complete_for((2, 5)))
        self.assertFalse(index.is_complete_for((2,)))
        self.assertEqual(
            index.changed_keyframes(
                keyframe_ids=(2, 5),
                current_poses={2: _pose(), 5: _pose(1.0)},
            ),
            (),
        )
        self.assertEqual(
            index.changed_keyframes(
                keyframe_ids=(2, 5),
                current_poses={2: _pose(), 5: _pose(1.000000001)},
            ),
            (5,),
        )

    def test_invalid_pose_and_mismatched_batch_fail(self):
        index = MapContributionIndex()
        with self.assertRaisesRegex(ValueError, "Invalid"):
            index.record(
                keyframe_id=1,
                pose_world_camera=np.eye(3),
                tile_keys=(),
            )
        with self.assertRaisesRegex(ValueError, "sets differ"):
            index.record_many(
                poses_world_camera={1: _pose()},
                tile_keys_by_keyframe={2: ()},
            )
        self.assertEqual(len(index), 0)


if __name__ == "__main__":
    unittest.main()
