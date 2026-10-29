"""Pinned DINOv2, SALAD and SONATA inference backend.

Only model definitions are imported from the pinned third-party source.  The
online preprocessing, projection, caching and safety checks are implemented
here so the offline MPRF scripts' dataset paths and hard-coded calibration do
not leak into the robot runtime.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import MprfRuntimeConfiguration
from .contracts import KeyframePacket, PoseFeatureSet
from .geometry import normalized_rows, normalized_vector, transform_points


def _load_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module specification from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _torch_load_state(torch, path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _preprocess_image(image_bgr: np.ndarray, size: int) -> np.ndarray:
    """Match MPRF contrast enhancement without torchvision or file I/O."""

    image = np.asarray(image_bgr, dtype=np.uint8)
    gray = np.clip(
        0.114 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.299 * image[:, :, 2],
        0.0,
        255.0,
    ).astype(np.uint8)
    enhanced = ImageOps.equalize(Image.fromarray(gray, mode="L"))
    rgb = enhanced.convert("RGB").resize(
        (int(size), int(size)), resample=Image.Resampling.BILINEAR
    )
    value = np.asarray(rgb, dtype=np.float32) / 255.0
    value = (value - np.asarray([0.485, 0.456, 0.406], dtype=np.float32))
    value /= np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return np.ascontiguousarray(value.transpose(2, 0, 1)[None])


def _deterministic_voxel_sample(
    points: np.ndarray, *, voxel_size_m: float, maximum_points: int
) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    value = value[np.isfinite(value).all(axis=1)]
    if len(value) and float(voxel_size_m) > 0.0:
        voxels = np.floor(value / float(voxel_size_m)).astype(np.int64)
        _, first = np.unique(voxels, axis=0, return_index=True)
        value = value[np.sort(first)]
    if len(value) > int(maximum_points):
        indices = np.linspace(
            0, len(value) - 1, int(maximum_points), dtype=np.int64
        )
        value = value[indices]
    return np.ascontiguousarray(value)


def _sonata_input(torch, points: np.ndarray, *, voxel_size_m: float):
    """Build SONATA's Point input with deterministic voxel representatives."""

    original = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    center = np.asarray(
        [
            0.5 * (original[:, 0].min() + original[:, 0].max()),
            0.5 * (original[:, 1].min() + original[:, 1].max()),
            original[:, 2].min(),
        ],
        dtype=np.float32,
    )
    centered = original - center
    grid = np.floor(centered / float(voxel_size_m)).astype(np.int64)
    grid -= np.min(grid, axis=0)
    _, first, inverse_unique = np.unique(
        grid, axis=0, return_index=True, return_inverse=True
    )
    selected_order = np.argsort(first, kind="stable")
    selected = first[selected_order]
    unique_to_selected = np.empty(len(first), dtype=np.int64)
    unique_to_selected[selected_order] = np.arange(len(first), dtype=np.int64)
    inverse = unique_to_selected[inverse_unique]
    selected_coord = centered[selected]
    selected_grid = grid[selected]
    zeros = np.zeros_like(selected_coord, dtype=np.float32)
    features = np.concatenate([selected_coord, zeros, zeros], axis=1)
    return {
        "coord": torch.from_numpy(np.ascontiguousarray(selected_coord)),
        "grid_coord": torch.from_numpy(
            np.ascontiguousarray(selected_grid)
        ).long(),
        "color": torch.from_numpy(np.ascontiguousarray(zeros)),
        "inverse": torch.from_numpy(np.ascontiguousarray(inverse)).long(),
        "offset": torch.tensor([len(selected)], dtype=torch.long),
        "feat": torch.from_numpy(np.ascontiguousarray(features)),
    }


def _unpool_sonata_features(point):
    # This is the feature reconstruction used by SONATA's official example.
    import torch

    for _ in range(2):
        if "pooling_parent" not in point.keys() or "pooling_inverse" not in point.keys():
            raise RuntimeError("SONATA output omitted the expected pooling trace")
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        # Preserve the parent-scale representation as in SONATA's official
        # dense-feature example; this is the multimodal descriptor MPRF was
        # evaluated with, not merely the final decoder-free feature.
        parent.feat = torch.cat(
            [parent.feat, point.feat[inverse]], dim=-1
        )
        point = parent
    while "pooling_parent" in point.keys():
        if "pooling_inverse" not in point.keys():
            raise RuntimeError("SONATA pooling trace is incomplete")
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent
    return point


class TorchMprfFeatureBackend:
    """One process owns all three models and exposes immutable NumPy output."""

    def __init__(self, configuration: MprfRuntimeConfiguration) -> None:
        configuration.validate_assets()
        self.configuration = configuration
        try:
            import torch
            import torch.nn as nn
        except ImportError as exception:
            raise RuntimeError("MPRF inference requires PyTorch") from exception
        self.torch = torch
        self.device = torch.device(configuration.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"MPRF device {self.device} requested but CUDA is unavailable"
            )
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        aggregator_module = _load_module_from_file(
            "_t3_mprf_salad_aggregator",
            configuration.salad_source
            / "models"
            / "aggregators"
            / "salad.py",
        )
        salad_aggregator_class = aggregator_module.SALAD

        dino_source = str(configuration.dinov2_source)
        salad_dino = torch.hub.load(
            dino_source,
            "dinov2_vitb14",
            source="local",
            pretrained=False,
        )

        class SaladBackbone(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, value):
                batch, _, height, width = value.shape
                tokens = self.model.prepare_tokens_with_masks(value)
                for block in self.model.blocks:
                    tokens = block(tokens)
                tokens = self.model.norm(tokens)
                token = tokens[:, 0]
                patches = tokens[:, 1:]
                patches = patches.reshape(
                    batch, height // 14, width // 14, 768
                ).permute(0, 3, 1, 2)
                return patches, token

        class SaladInference(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
                self.aggregator = salad_aggregator_class(
                    num_channels=768,
                    num_clusters=64,
                    cluster_dim=128,
                    token_dim=256,
                )

            def forward(self, value):
                return self.aggregator(self.backbone(value))

        self.salad_model = SaladInference(SaladBackbone(salad_dino))
        salad_state = _torch_load_state(torch, configuration.salad_weights)
        if not isinstance(salad_state, dict) or "state_dict" in salad_state:
            raise RuntimeError(
                "the selected SALAD checkpoint is not the pinned flat pretrained state"
            )
        self.salad_model.load_state_dict(salad_state, strict=True)
        self.salad_model.to(self.device).eval()

        self.dino_model = torch.hub.load(
            dino_source,
            "dinov2_vitb14_reg",
            source="local",
            pretrained=False,
        )
        dino_state = _torch_load_state(torch, configuration.dino_weights)
        self.dino_model.load_state_dict(dino_state, strict=True)
        self.dino_model.to(self.device).eval()

        sonata_root = str(configuration.sonata_source)
        if sonata_root not in sys.path:
            sys.path.insert(0, sonata_root)
        sonata = importlib.import_module("sonata")
        loaded_from = Path(sonata.__file__).resolve()
        if configuration.sonata_source not in loaded_from.parents:
            raise ImportError(
                f"unexpected SONATA module was imported from {loaded_from}"
            )
        self.sonata = sonata
        self.sonata_model = sonata.model.load(
            str(configuration.sonata_weights),
            custom_config={
                "enc_patch_size": [512, 512, 512, 512, 512],
                "enable_flash": False,
                "shuffle_orders": False,
                "upcast_attention": True,
                "upcast_softmax": True,
            },
        ).to(self.device).eval()

    def _autocast(self):
        if self.device.type != "cuda" or not self.configuration.use_amp:
            return nullcontext()
        return self.torch.autocast(
            device_type="cuda", dtype=self.torch.float16
        )

    def _tensor(self, image_bgr: np.ndarray, size: int):
        value = _preprocess_image(image_bgr, size)
        return self.torch.from_numpy(value).to(self.device, non_blocking=True)

    def extract_retrieval(
        self, image_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        salad_input = self._tensor(
            image_bgr, self.configuration.salad_image_size
        )
        dino_input = self._tensor(
            image_bgr, self.configuration.dino_image_size
        )
        with torch.inference_mode(), self._autocast():
            salad = self.salad_model(salad_input)
            layers = self.dino_model.get_intermediate_layers(dino_input, n=3)
            dino = torch.cat(layers, dim=-1).mean(dim=1)
            salad = torch.nn.functional.normalize(salad.float(), dim=-1)
            dino = torch.nn.functional.normalize(dino.float(), dim=-1)
        return (
            normalized_vector(salad[0].cpu().numpy()),
            normalized_vector(dino[0].cpu().numpy()),
        )

    def _sonata_features(self, points_sensor: np.ndarray) -> np.ndarray:
        torch = self.torch
        point = _sonata_input(
            torch,
            points_sensor,
            voxel_size_m=self.configuration.pose_voxel_size_m,
        )
        for key, value in tuple(point.items()):
            if isinstance(value, torch.Tensor):
                point[key] = value.to(self.device, non_blocking=True)
        with torch.inference_mode(), self._autocast():
            encoded = self.sonata_model(point)
            reconstructed = _unpool_sonata_features(encoded)
            if "inverse" not in reconstructed.keys():
                raise RuntimeError("SONATA output omitted the input inverse map")
            features = reconstructed.feat[reconstructed.inverse]
            features = torch.nn.functional.normalize(features.float(), dim=-1)
        result = features.cpu().numpy().astype(np.float32, copy=False)
        if len(result) != len(points_sensor):
            raise RuntimeError(
                "SONATA reconstructed feature count does not match input points"
            )
        return result

    def _dino_patch_features(self, image_bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
        size = self.configuration.pose_image_size
        image = self._tensor(image_bgr, size)
        with torch.inference_mode(), self._autocast():
            layers = self.dino_model.get_intermediate_layers(image, n=3)
            features = torch.cat(layers, dim=-1)
            features = torch.nn.functional.normalize(features.float(), dim=-1)
        result = features[0].cpu().numpy().astype(np.float32, copy=False)
        expected = (size // 14) ** 2
        if result.shape[0] != expected:
            raise RuntimeError(
                f"DINO produced {result.shape[0]} patches; expected {expected}"
            )
        return result

    def extract_pose_features(self, packet: KeyframePacket) -> PoseFeatureSet:
        points = _deterministic_voxel_sample(
            packet.points_sensor,
            voxel_size_m=self.configuration.pose_voxel_size_m,
            maximum_points=self.configuration.max_pose_points,
        )
        if len(points) < 3:
            return PoseFeatureSet(
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
            )
        sonata = self._sonata_features(points)
        dino_grid = self._dino_patch_features(packet.image_bgr)

        points_camera = transform_points(
            packet.transform_camera_sensor, points
        )
        image_height, image_width = packet.image_bgr.shape[:2]
        camera = packet.camera_matrix.copy()
        camera[0, :] *= image_width / float(packet.calibration_width)
        camera[1, :] *= image_height / float(packet.calibration_height)
        depth = points_camera[:, 2]
        u = camera[0, 0] * points_camera[:, 0] / np.maximum(depth, 1.0e-9)
        u += camera[0, 2]
        v = camera[1, 1] * points_camera[:, 1] / np.maximum(depth, 1.0e-9)
        v += camera[1, 2]
        valid = (
            (depth > 1.0e-6)
            & (u >= 0.0)
            & (u < image_width)
            & (v >= 0.0)
            & (v < image_height)
        )
        if not np.any(valid):
            return PoseFeatureSet(
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
            )
        grid_size = self.configuration.pose_image_size // 14
        patch_x = np.floor(u[valid] / image_width * grid_size).astype(np.int64)
        patch_y = np.floor(v[valid] / image_height * grid_size).astype(np.int64)
        patch_x = np.clip(patch_x, 0, grid_size - 1)
        patch_y = np.clip(patch_y, 0, grid_size - 1)
        dino = dino_grid[patch_y * grid_size + patch_x]
        dino = normalized_rows(dino)
        sonata = normalized_rows(sonata[valid])
        dino_weight = np.sqrt(self.configuration.dino_feature_weight)
        sonata_weight = np.sqrt(self.configuration.sonata_feature_weight)
        combined = np.concatenate(
            [dino * dino_weight, sonata * sonata_weight], axis=1
        )
        combined = normalized_rows(combined)
        return PoseFeatureSet(points[valid], combined)

    def metadata(self) -> dict[str, object]:
        torch = self.torch
        result = {
            "backend": "dinov2_salad_sonata",
            "device": str(self.device),
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "amp": bool(self.configuration.use_amp),
        }
        if self.device.type == "cuda":
            result["device_name"] = torch.cuda.get_device_name(self.device)
        return result
