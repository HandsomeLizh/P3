from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def validated_pose_matrix(pose: np.ndarray, *, name: str = "pose") -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = value[:3, :3]
    orthogonality_error = np.linalg.norm(
        rotation.T @ rotation - np.eye(3), ord="fro"
    )
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > 1e-3 or determinant <= 0.0:
        raise ValueError(
            f"{name} rotation is invalid: error={orthogonality_error:.3e}, "
            f"det={determinant:.6f}"
        )
    return value.copy()


def pose_matrix(
    position_xyz: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    position = np.asarray(position_xyz, dtype=np.float64).reshape(3)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("Position and quaternion must be finite")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("Quaternion norm is zero")

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    result[:3, 3] = position
    return result


def inverse_pose(pose: np.ndarray) -> np.ndarray:
    value = validated_pose_matrix(pose)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -(result[:3, :3] @ value[:3, 3])
    return result


def interpolate_pose(first: np.ndarray, second: np.ndarray, ratio: float) -> np.ndarray:
    first_pose = validated_pose_matrix(first, name="first pose")
    second_pose = validated_pose_matrix(second, name="second pose")
    alpha = float(ratio)
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"Interpolation ratio must be in [0, 1], got {ratio}")
    if alpha == 0.0:
        return first_pose
    if alpha == 1.0:
        return second_pose

    rotations = Rotation.from_matrix(
        np.stack([first_pose[:3, :3], second_pose[:3, :3]], axis=0)
    )
    rotation = Slerp([0.0, 1.0], rotations)([alpha]).as_matrix()[0]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = (
        (1.0 - alpha) * first_pose[:3, 3]
        + alpha * second_pose[:3, 3]
    )
    return result


def child_pose_to_camera_pose(
    pose_parent_child: np.ndarray,
    transform_camera_child: np.ndarray,
) -> np.ndarray:
    """Convert ``T_parent_child`` to ``T_parent_camera``.

    ``transform_camera_child`` follows the calibration convention
    ``p_camera = T_camera_child @ p_child``.  Therefore the camera pose in the
    parent frame is ``T_parent_child @ inverse(T_camera_child)``.
    """

    parent_child = validated_pose_matrix(
        pose_parent_child, name="parent-to-child pose"
    )
    camera_child = validated_pose_matrix(
        transform_camera_child, name="child-to-camera calibration"
    )
    return parent_child @ inverse_pose(camera_child)


def relative_pose(reference: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return inverse_pose(reference) @ validated_pose_matrix(pose)


def pose_change_norm(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    delta = inverse_pose(first) @ validated_pose_matrix(second)
    translation_m = float(np.linalg.norm(delta[:3, 3]))
    rotation_rad = float(Rotation.from_matrix(delta[:3, :3]).magnitude())
    return translation_m, rotation_rad
