from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .pose_math import inverse_pose, validated_pose_matrix
from .types import FusionUpdate, PoseEstimate


class FusionDependencyError(RuntimeError):
    pass


class FusionInputError(ValueError):
    pass


def _positive_triplet(value, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain three positive finite values")
    return result


class GtsamGpWnoaFusion:
    """Fixed-lag Pose3 fusion with a continuous-time WNOA motion prior.

    Each support state contains ``Pose3`` and a six-dimensional body tangent
    velocity ordered as GTSAM's ``[wx, wy, wz, vx, vy, vz]``.  The GP process
    covariance follows the standard WNOA block form used by gnssFGO:

    ``Q = [[dt^3/3 Qc, dt^2/2 Qc], [dt^2/2 Qc, dt Qc]]``.

    Only relative VO and LIO pose measurements are fused.  Raw IMU is not
    accepted here because a FAST-LIO2 odometry input has already consumed it.
    """

    def __init__(
        self,
        *,
        lag_sec: float = 10.0,
        enable_gp_prior: bool = True,
        solver: str = "incremental",
        qc_rotation=(1.0, 1.0, 2.0),
        qc_translation=(2.0, 2.0, 2.0),
        vo_rotation_sigma_deg=(1.5, 1.5, 2.5),
        vo_translation_sigma_m=(0.08, 0.08, 0.12),
        lio_rotation_sigma_deg=(0.5, 0.5, 0.8),
        lio_translation_sigma_m=(0.03, 0.03, 0.05),
        robust_huber_k: float = 1.5,
        minimum_dt_sec: float = 1e-4,
    ) -> None:
        try:
            import gtsam
            import gtsam_unstable
        except ImportError as exception:
            raise FusionDependencyError(
                "fusion_backend='gtsam_gp' requires Python modules gtsam "
                "and gtsam_unstable"
            ) from exception

        self.gtsam = gtsam
        self.gtsam_unstable = gtsam_unstable
        self.lag_sec = float(lag_sec)
        self.enable_gp_prior = bool(enable_gp_prior)
        self.solver = str(solver).strip().lower()
        self.minimum_dt_sec = float(minimum_dt_sec)
        self.robust_huber_k = float(robust_huber_k)
        if self.lag_sec <= 0.0:
            raise ValueError("fusion_lag_sec must be positive")
        if self.minimum_dt_sec <= 0.0:
            raise ValueError("minimum_dt_sec must be positive")
        if self.solver not in {"incremental", "batch"}:
            raise ValueError("fusion solver must be incremental or batch")

        self.qc = np.r_[
            _positive_triplet(qc_rotation, name="qc_rotation"),
            _positive_triplet(qc_translation, name="qc_translation"),
        ]
        self.vo_sigmas = np.r_[
            np.deg2rad(
                _positive_triplet(
                    vo_rotation_sigma_deg, name="vo_rotation_sigma_deg"
                )
            ),
            _positive_triplet(
                vo_translation_sigma_m, name="vo_translation_sigma_m"
            ),
        ]
        self.lio_sigmas = np.r_[
            np.deg2rad(
                _positive_triplet(
                    lio_rotation_sigma_deg, name="lio_rotation_sigma_deg"
                )
            ),
            _positive_triplet(
                lio_translation_sigma_m, name="lio_translation_sigma_m"
            ),
        ]

        self._smoother = self._make_smoother()
        self._first_timestamp: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._next_index = 0
        self._references: dict[str, np.ndarray] = {}
        self._source_alignments: dict[str, object] = {}
        self._last_measurement: dict[str, tuple[int, float, object]] = {}
        self._last_estimate = None
        self._active_indices: set[int] = set()
        self._state_timestamps: dict[int, float] = {}
        # Keep callbacks alive while their Python CustomFactors may still be
        # referenced by the fixed-lag smoother.  The deque is bounded by a
        # conservative rate-independent count, not by process lifetime.
        self._factor_callbacks = deque(maxlen=4096)

    def _make_smoother(self):
        if self.solver == "batch":
            params = self.gtsam.LevenbergMarquardtParams()
            params.setVerbosityLM("SILENT")
            return self.gtsam_unstable.BatchFixedLagSmoother(
                self.lag_sec, params
            )
        params = self.gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.01)
        params.relinearizeSkip = 1
        return self.gtsam_unstable.IncrementalFixedLagSmoother(
            self.lag_sec, params
        )

    def _x(self, index: int) -> int:
        return self.gtsam.symbol("x", int(index))

    def _v(self, index: int) -> int:
        return self.gtsam.symbol("v", int(index))

    def _pose3(self, pose: np.ndarray):
        value = validated_pose_matrix(pose)
        return self.gtsam.Pose3(value)

    def _normalised_measurement(self, source: str, pose: np.ndarray):
        value = validated_pose_matrix(pose, name=f"{source} pose")
        reference = self._references.get(source)
        if reference is None:
            reference = value.copy()
            self._references[source] = reference
        return self._pose3(inverse_pose(reference) @ value)

    def _measurement_noise(self, sigmas: np.ndarray):
        base = self.gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        if self.robust_huber_k <= 0.0:
            return base
        return self.gtsam.noiseModel.Robust.Create(
            self.gtsam.noiseModel.mEstimator.Huber.Create(
                self.robust_huber_k
            ),
            base,
        )

    def _gp_noise(self, dt: float):
        qc = np.diag(self.qc)
        covariance = np.block(
            [
                [(dt ** 3 / 3.0) * qc, (dt ** 2 / 2.0) * qc],
                [(dt ** 2 / 2.0) * qc, dt * qc],
            ]
        )
        covariance += np.eye(12, dtype=np.float64) * 1e-12
        return self.gtsam.noiseModel.Gaussian.Covariance(covariance)

    def _make_gp_factor(self, first_index: int, second_index: int, dt: float):
        gtsam = self.gtsam
        first_pose_key = self._x(first_index)
        first_velocity_key = self._v(first_index)
        second_pose_key = self._x(second_index)
        second_velocity_key = self._v(second_index)

        def evaluate(first_pose, first_velocity, second_pose, second_velocity):
            relative = first_pose.between(second_pose)
            tangent = np.asarray(gtsam.Pose3.Logmap(relative)).reshape(6)
            inverse_right_jacobian = np.asarray(
                gtsam.Pose3.LogmapDerivative(relative)
            ).reshape(6, 6)
            return np.r_[
                tangent - first_velocity * dt,
                inverse_right_jacobian @ second_velocity - first_velocity,
            ]

        def numerical_pose_jacobian(function, pose):
            epsilon = 1e-6
            jacobian = np.empty((12, 6), dtype=np.float64)
            for column in range(6):
                delta = np.zeros(6, dtype=np.float64)
                delta[column] = epsilon
                jacobian[:, column] = (
                    function(pose.retract(delta))
                    - function(pose.retract(-delta))
                ) / (2.0 * epsilon)
            return jacobian

        def error_function(this_factor, values, jacobians):
            del this_factor
            first_pose = values.atPose3(first_pose_key)
            first_velocity = np.asarray(
                values.atVector(first_velocity_key)
            ).reshape(6)
            second_pose = values.atPose3(second_pose_key)
            second_velocity = np.asarray(
                values.atVector(second_velocity_key)
            ).reshape(6)
            error = evaluate(
                first_pose,
                first_velocity,
                second_pose,
                second_velocity,
            )
            if jacobians is not None:
                jacobians[0] = numerical_pose_jacobian(
                    lambda pose: evaluate(
                        pose,
                        first_velocity,
                        second_pose,
                        second_velocity,
                    ),
                    first_pose,
                )
                jacobians[1] = np.vstack(
                    (-dt * np.eye(6), -np.eye(6))
                )
                jacobians[2] = numerical_pose_jacobian(
                    lambda pose: evaluate(
                        first_pose,
                        first_velocity,
                        pose,
                        second_velocity,
                    ),
                    second_pose,
                )
                relative = first_pose.between(second_pose)
                inverse_right_jacobian = np.asarray(
                    gtsam.Pose3.LogmapDerivative(relative)
                ).reshape(6, 6)
                jacobians[3] = np.vstack(
                    (np.zeros((6, 6)), inverse_right_jacobian)
                )
            return error

        self._factor_callbacks.append(error_function)
        return gtsam.CustomFactor(
            self._gp_noise(dt),
            [
                first_pose_key,
                first_velocity_key,
                second_pose_key,
                second_velocity_key,
            ],
            error_function,
        )

    def _relative_initialisation(
        self,
        *,
        previous_index: int,
        previous_pose,
        previous_velocity: np.ndarray,
        dt: float,
        measurements: dict[str, object],
    ):
        for source in ("lio", "vo"):
            previous = self._last_measurement.get(source)
            if (
                source in measurements
                and previous is not None
                and previous[0] == previous_index
            ):
                delta = previous[2].between(measurements[source])
                return previous_pose.compose(delta)
        return previous_pose.compose(
            self.gtsam.Pose3.Expmap(previous_velocity * dt)
        )

    def add_measurement(
        self,
        *,
        timestamp: float,
        vo_pose: Optional[np.ndarray],
        lio_pose: Optional[np.ndarray],
    ) -> FusionUpdate:
        sensor_time = float(timestamp)
        if not np.isfinite(sensor_time):
            raise FusionInputError("fusion timestamp must be finite")
        if vo_pose is None and lio_pose is None:
            raise FusionInputError("fusion requires at least one pose source")
        if self._last_timestamp is not None:
            dt = sensor_time - self._last_timestamp
            if dt < self.minimum_dt_sec:
                raise FusionInputError(
                    "fusion timestamps must be strictly increasing: "
                    f"dt={dt:.9f}s"
                )
        else:
            dt = 0.0

        local_measurements = {}
        if vo_pose is not None:
            local_measurements["vo"] = self._normalised_measurement(
                "vo", vo_pose
            )
        if lio_pose is not None:
            local_measurements["lio"] = self._normalised_measurement(
                "lio", lio_pose
            )

        # Existing modality streams are already expressed in the fused frame.
        # A modality that first appears later is aligned only after this
        # state's initial pose has been predicted from the currently available
        # trajectory; otherwise its own zero origin would pull the graph back
        # to the beginning of the run.
        measurements = {
            source: self._source_alignments[source].compose(measurement)
            for source, measurement in local_measurements.items()
            if source in self._source_alignments
        }

        if self._first_timestamp is None:
            self._first_timestamp = sensor_time
        relative_time = sensor_time - self._first_timestamp
        index = self._next_index
        graph = self.gtsam.NonlinearFactorGraph()
        values = self.gtsam.Values()
        timestamps = self.gtsam_unstable.FixedLagSmootherKeyTimestampMap()

        if index == 0:
            initial_pose = self.gtsam.Pose3()
            initial_velocity = np.zeros(6, dtype=np.float64)
            values.insert(self._x(index), initial_pose)
            graph.add(
                self.gtsam.PriorFactorPose3(
                    self._x(index),
                    initial_pose,
                    self.gtsam.noiseModel.Diagonal.Sigmas(
                        np.r_[np.full(3, 1e-4), np.full(3, 1e-4)]
                    ),
                )
            )
            if self.enable_gp_prior:
                values.insert(self._v(index), initial_velocity)
                graph.add(
                    self.gtsam.PriorFactorVector(
                        self._v(index),
                        initial_velocity,
                        self.gtsam.noiseModel.Diagonal.Sigmas(
                            np.r_[np.full(3, 1.5), np.full(3, 2.0)]
                        ),
                    )
                )
        else:
            previous_index = index - 1
            if self._last_estimate is None:
                raise RuntimeError("fusion estimate state is unavailable")
            previous_pose = self._last_estimate.atPose3(
                self._x(previous_index)
            )
            previous_velocity = (
                np.asarray(
                    self._last_estimate.atVector(self._v(previous_index))
                ).reshape(6)
                if self.enable_gp_prior
                else np.zeros(6, dtype=np.float64)
            )
            initial_pose = self._relative_initialisation(
                previous_index=previous_index,
                previous_pose=previous_pose,
                previous_velocity=previous_velocity,
                dt=dt,
                measurements=measurements,
            )
            values.insert(self._x(index), initial_pose)
            if self.enable_gp_prior:
                initial_velocity = np.asarray(
                    self.gtsam.Pose3.Logmap(
                        previous_pose.between(initial_pose)
                    )
                ).reshape(6) / dt
                values.insert(self._v(index), initial_velocity)
                graph.add(self._make_gp_factor(previous_index, index, dt))

        for source, local_measurement in local_measurements.items():
            if source not in self._source_alignments:
                self._source_alignments[source] = initial_pose.compose(
                    local_measurement.inverse()
                )
            measurements[source] = self._source_alignments[source].compose(
                local_measurement
            )

        timestamps.insert((self._x(index), float(relative_time)))
        if self.enable_gp_prior:
            timestamps.insert((self._v(index), float(relative_time)))

        for source, measurement in measurements.items():
            sigmas = self.lio_sigmas if source == "lio" else self.vo_sigmas
            previous = self._last_measurement.get(source)
            if (
                previous is not None
                and previous[0] in self._active_indices
                and sensor_time - previous[1] < self.lag_sec
            ):
                graph.add(
                    self.gtsam.BetweenFactorPose3(
                        self._x(previous[0]),
                        self._x(index),
                        previous[2].between(measurement),
                        self._measurement_noise(sigmas),
                    )
                )
            elif index > 0:
                # A modality returning after more than one fixed-lag window
                # cannot reference its marginalised old key.  Re-anchor it
                # softly, then resume relative factors on subsequent states.
                graph.add(
                    self.gtsam.PriorFactorPose3(
                        self._x(index),
                        measurement,
                        self._measurement_noise(sigmas * 3.0),
                    )
                )

        self._smoother.update(graph, values, timestamps)
        estimate = self._smoother.calculateEstimate()
        current_pose3 = estimate.atPose3(self._x(index))
        current_pose = np.asarray(current_pose3.matrix(), dtype=np.float64)
        velocity = (
            np.asarray(estimate.atVector(self._v(index))).reshape(6)
            if self.enable_gp_prior
            else None
        )

        self._state_timestamps[index] = sensor_time
        active_poses = {}
        active_indices = set()
        for candidate in tuple(self._state_timestamps):
            key = self._x(candidate)
            if estimate.exists(key):
                active_indices.add(candidate)
                active_poses[candidate] = np.asarray(
                    estimate.atPose3(key).matrix(), dtype=np.float64
                )
            else:
                # Once a key has left the fixed-lag result it cannot become
                # active again.  Removing it bounds this bookkeeping by the
                # real time window even for irregular input frame rates.
                self._state_timestamps.pop(candidate, None)

        for source, measurement in measurements.items():
            self._last_measurement[source] = (
                index,
                sensor_time,
                measurement,
            )
        self._active_indices = active_indices
        self._last_estimate = estimate
        self._last_timestamp = sensor_time
        self._next_index += 1

        return FusionUpdate(
            estimate=PoseEstimate(
                timestamp=sensor_time,
                pose=current_pose,
                source="vo_lio_gtsam_gp",
                velocity_body=velocity,
                diagnostics={
                    "gp_prior_enabled": self.enable_gp_prior,
                    "active_states": len(active_poses),
                    "vo_factor_added": "vo" in measurements,
                    "lio_factor_added": "lio" in measurements,
                },
            ),
            measurement_index=index,
            active_poses=active_poses,
        )
