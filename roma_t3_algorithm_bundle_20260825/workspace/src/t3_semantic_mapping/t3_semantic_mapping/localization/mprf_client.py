"""Non-blocking client for the isolated MPRF inference service.

All compression and HTTP work happens on one daemon worker.  The ROS callback
only copies a bounded keyframe snapshot and polls completed results.  A
service failure therefore cannot stop local VO/LIO odometry or map updates.
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np

from .loop_constraints import LoopCandidate
from .pose_math import validated_pose_matrix


MPRF_PROTOCOL_VERSION = 1
_TOKEN_ENV_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class MprfClientConfiguration:
    enabled: bool = False
    endpoint: str = "http://127.0.0.1:8765"
    request_timeout_sec: float = 20.0
    queue_size: int = 2
    transient_retries: int = 2
    auth_token_env: str = "MPRF_SERVICE_TOKEN"

    def __post_init__(self) -> None:
        endpoint = str(self.endpoint).strip().rstrip("/")
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "MPRF endpoint must be an http(s) origin without credentials, "
                "query or fragment"
            )
        timeout = float(self.request_timeout_sec)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("MPRF request timeout must be positive")
        queue_size = int(self.queue_size)
        retries = int(self.transient_retries)
        if queue_size <= 0 or retries < 0:
            raise ValueError("MPRF queue must be positive and retries non-negative")
        token_env = str(self.auth_token_env).strip()
        if token_env and not _TOKEN_ENV_PATTERN.fullmatch(token_env):
            raise ValueError("MPRF auth token environment variable name is invalid")
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "request_timeout_sec", timeout)
        object.__setattr__(self, "queue_size", queue_size)
        object.__setattr__(self, "transient_retries", retries)
        object.__setattr__(self, "auth_token_env", token_env)


@dataclass(frozen=True)
class MprfKeyframeRequest:
    session_id: str
    node_id: int
    timestamp_ns: int
    image_bgr: np.ndarray
    points_sensor: np.ndarray
    camera_matrix: np.ndarray
    transform_camera_sensor: np.ndarray
    calibration_width: int
    calibration_height: int

    def __post_init__(self) -> None:
        session = str(self.session_id).strip()
        node = int(self.node_id)
        timestamp = int(self.timestamp_ns)
        if not session or node < 0 or timestamp <= 0:
            raise ValueError("invalid MPRF request identity")
        image = np.asarray(self.image_bgr)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("MPRF image must be HxWx3 uint8 BGR")
        points = np.asarray(self.points_sensor, dtype=np.float32).reshape(-1, 3)
        if not np.isfinite(points).all():
            raise ValueError("MPRF points contain non-finite values")
        camera = np.asarray(self.camera_matrix, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(camera).all() or min(camera[0, 0], camera[1, 1]) <= 0.0:
            raise ValueError("MPRF camera matrix is invalid")
        transform = validated_pose_matrix(
            self.transform_camera_sensor,
            name="MPRF camera-sensor transform",
        )
        width = int(self.calibration_width)
        height = int(self.calibration_height)
        if width <= 0 or height <= 0:
            raise ValueError("MPRF calibration dimensions must be positive")
        image = np.ascontiguousarray(image.copy())
        points = np.ascontiguousarray(points.copy())
        camera = np.ascontiguousarray(camera.copy())
        transform = np.ascontiguousarray(transform.copy())
        for value in (image, points, camera, transform):
            value.setflags(write=False)
        object.__setattr__(self, "session_id", session)
        object.__setattr__(self, "node_id", node)
        object.__setattr__(self, "timestamp_ns", timestamp)
        object.__setattr__(self, "image_bgr", image)
        object.__setattr__(self, "points_sensor", points)
        object.__setattr__(self, "camera_matrix", camera)
        object.__setattr__(self, "transform_camera_sensor", transform)
        object.__setattr__(self, "calibration_width", width)
        object.__setattr__(self, "calibration_height", height)


@dataclass(frozen=True)
class MprfClientResult:
    query_node: int
    query_timestamp_ns: int
    status: str
    candidates: tuple[LoopCandidate, ...]
    diagnostics: dict[str, object]
    error: Optional[str] = None

    @property
    def candidate(self) -> Optional[LoopCandidate]:
        """Protocol-v1 compatibility view of the first proposal."""

        return self.candidates[0] if self.candidates else None


def encode_mprf_request(request: MprfKeyframeRequest) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        protocol_version=np.asarray(MPRF_PROTOCOL_VERSION, dtype=np.int32),
        session_id=np.asarray(request.session_id),
        node_id=np.asarray(request.node_id, dtype=np.int64),
        timestamp_ns=np.asarray(request.timestamp_ns, dtype=np.int64),
        image_bgr=request.image_bgr,
        points_sensor=request.points_sensor,
        camera_matrix=request.camera_matrix,
        transform_camera_sensor=request.transform_camera_sensor,
        calibration_size=np.asarray(
            [request.calibration_width, request.calibration_height],
            dtype=np.int32,
        ),
    )
    return stream.getvalue()


class MprfAsyncClient:
    """Latest-first bounded queue with validated result delivery."""

    _STOP = object()

    def __init__(
        self,
        configuration: MprfClientConfiguration,
        *,
        session_id: Optional[str] = None,
    ) -> None:
        self.configuration = configuration
        self.session_id = str(session_id or uuid.uuid4().hex)
        self._queue: queue.Queue = queue.Queue(
            maxsize=configuration.queue_size
        )
        self._results: deque[MprfClientResult] = deque()
        self._result_lock = threading.Lock()
        self._closed = False
        self.submitted_count = 0
        self.dropped_count = 0
        self.completed_count = 0
        self.failed_count = 0
        self._thread = None
        if configuration.enabled:
            self._thread = threading.Thread(
                target=self._run,
                name="mprf-http-worker",
                daemon=True,
            )
            self._thread.start()

    def submit(
        self,
        *,
        node_id: int,
        timestamp_ns: int,
        image_bgr: np.ndarray,
        points_sensor: np.ndarray,
        camera_matrix: np.ndarray,
        transform_camera_sensor: np.ndarray,
        calibration_width: int,
        calibration_height: int,
    ) -> bool:
        if not self.configuration.enabled or self._closed:
            return False
        request = MprfKeyframeRequest(
            session_id=self.session_id,
            node_id=node_id,
            timestamp_ns=timestamp_ns,
            image_bgr=image_bgr,
            points_sensor=points_sensor,
            camera_matrix=camera_matrix,
            transform_camera_sensor=transform_camera_sensor,
            calibration_width=calibration_width,
            calibration_height=calibration_height,
        )
        try:
            self._queue.put_nowait(request)
        except queue.Full:
            try:
                discarded = self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                discarded = None
            if discarded is self._STOP:
                return False
            self.dropped_count += 1
            try:
                self._queue.put_nowait(request)
            except queue.Full:
                self.dropped_count += 1
                return False
        self.submitted_count += 1
        return True

    def poll(self) -> tuple[MprfClientResult, ...]:
        with self._result_lock:
            values = tuple(self._results)
            self._results.clear()
        return values

    def _deliver(self, result: MprfClientResult) -> None:
        with self._result_lock:
            self._results.append(result)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/x-npz",
            "Accept": "application/json",
        }
        token_env = self.configuration.auth_token_env
        token = os.environ.get(token_env, "").strip() if token_env else ""
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _post(self, request: MprfKeyframeRequest) -> dict[str, object]:
        body = encode_mprf_request(request)
        http_request = Request(
            self.configuration.endpoint + "/v1/keyframes",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        with urlopen(
            http_request,
            timeout=self.configuration.request_timeout_sec,
        ) as response:
            response_body = response.read(2 * 1024 * 1024 + 1)
            if len(response_body) > 2 * 1024 * 1024:
                raise ValueError("MPRF response exceeds 2 MiB")
            value = json.loads(response_body.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("MPRF response is not a JSON object")
        return value

    def _parse_response(
        self,
        request: MprfKeyframeRequest,
        response: dict[str, object],
    ) -> MprfClientResult:
        version = int(response.get("protocol_version", -1))
        session = str(response.get("session_id", ""))
        query_node = int(response.get("query_node", -1))
        query_timestamp = int(response.get("query_timestamp_ns", -1))
        if version != MPRF_PROTOCOL_VERSION:
            raise ValueError(f"MPRF protocol mismatch: {version}")
        if (
            session != request.session_id
            or query_node != request.node_id
            or query_timestamp != request.timestamp_ns
        ):
            raise ValueError("MPRF response identity does not match its request")
        status = str(response.get("status", "")).strip()
        candidates = ()
        candidate_payload = response.get("candidate")
        if status == "candidate":
            if not isinstance(candidate_payload, dict):
                raise ValueError("MPRF candidate response omitted candidate data")
            payloads = response.get("candidates")
            if payloads is None:
                payloads = [candidate_payload]
            if (
                not isinstance(payloads, list)
                or not payloads
                or len(payloads) > 16
                or not all(isinstance(value, dict) for value in payloads)
            ):
                raise ValueError("MPRF candidates must be a non-empty bounded list")
            if payloads[0] != candidate_payload:
                raise ValueError("MPRF singular and plural candidate fields differ")
            parsed_candidates = []
            references = set()
            for payload in payloads:
                reference = int(payload["reference_node"])
                candidate_query = int(payload["query_node"])
                if (
                    reference < 0
                    or reference >= query_node
                    or candidate_query != query_node
                    or reference in references
                ):
                    raise ValueError("MPRF candidate node ordering is invalid")
                references.add(reference)
                score = float(payload["score"])
                if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError("MPRF candidate score is outside [0, 1]")
                pose_payload = payload.get("relative_pose_reference_query")
                pose = None
                if pose_payload is not None:
                    pose = validated_pose_matrix(
                        np.asarray(pose_payload, dtype=np.float64).reshape(4, 4),
                        name="MPRF coarse relative pose",
                    )
                mode = str(
                    payload.get(
                        "candidate_mode",
                        "multimodal_pose" if pose is not None else "retrieval_only",
                    )
                )
                if mode not in {"multimodal_pose", "retrieval_only"}:
                    raise ValueError("MPRF candidate mode is invalid")
                if mode == "multimodal_pose" and pose is None:
                    raise ValueError("MPRF multimodal-pose candidate omitted its pose")
                candidate_diagnostics = {
                    str(key): value
                    for key, value in payload.items()
                    if key not in {"relative_pose_reference_query"}
                }
                parsed_candidates.append(
                    LoopCandidate(
                        reference_node=reference,
                        query_node=query_node,
                        source=(
                            "mprf_dinov2_salad_sonata"
                            if pose is not None
                            else "mprf_salad_dinov2_retrieval"
                        ),
                        score=score,
                        coarse_relative_pose=pose,
                        diagnostics=candidate_diagnostics,
                    )
                )
            candidates = tuple(parsed_candidates)
        elif status not in {
            "stored",
            "duplicate",
            "insufficient_points",
        }:
            raise ValueError(f"unsupported MPRF response status {status!r}")
        diagnostics = {
            str(key): value
            for key, value in response.items()
            if key not in {"candidate", "candidates"}
        }
        return MprfClientResult(
            query_node=query_node,
            query_timestamp_ns=query_timestamp,
            status=status,
            candidates=candidates,
            diagnostics=diagnostics,
        )

    def _process(self, request: MprfKeyframeRequest) -> MprfClientResult:
        retries = self.configuration.transient_retries
        for attempt in range(retries + 1):
            try:
                return self._parse_response(request, self._post(request))
            except HTTPError as exception:
                transient = exception.code in {429, 502, 503, 504}
                if transient and attempt < retries:
                    time.sleep(0.25 * (2 ** attempt))
                    continue
                raise RuntimeError(f"MPRF HTTP {exception.code}") from exception
            except URLError as exception:
                if attempt < retries:
                    time.sleep(0.25 * (2 ** attempt))
                    continue
                raise RuntimeError(f"MPRF connection failed: {exception.reason}") from exception
            except (TimeoutError, OSError) as exception:
                if attempt < retries:
                    time.sleep(0.25 * (2 ** attempt))
                    continue
                raise RuntimeError(
                    f"MPRF transport failed: {exception}"
                ) from exception
        raise RuntimeError("MPRF retry loop ended unexpectedly")

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                try:
                    result = self._process(item)
                    self.completed_count += 1
                except Exception as exception:
                    self.failed_count += 1
                    result = MprfClientResult(
                        query_node=item.node_id,
                        query_timestamp_ns=item.timestamp_ns,
                        status="error",
                        candidates=(),
                        diagnostics={},
                        error=f"{type(exception).__name__}: {exception}",
                    )
                self._deliver(result)
            finally:
                self._queue.task_done()

    def close(self, *, join_timeout_sec: float = 1.0) -> None:
        if self._closed:
            return
        self._closed = True
        thread = self._thread
        if thread is None:
            return
        while True:
            try:
                discarded = self._queue.get_nowait()
                self._queue.task_done()
                if discarded is not self._STOP:
                    self.dropped_count += 1
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            return
        thread.join(timeout=max(0.0, float(join_timeout_sec)))
