#!/usr/bin/env python3
"""Run one real project-configured RoMa match without starting ROS 2."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path


def parse_args():
    workspace = Path(__file__).resolve().parents[1]
    default_roma = workspace / "third_party" / "RoMa"
    parser = argparse.ArgumentParser(
        description="Check RoMa model loading and one stereo/temporal match."
    )
    parser.add_argument("sequence_dir", type=Path)
    parser.add_argument("--roma-root", type=Path, default=default_roma)
    parser.add_argument(
        "--pair",
        choices=("stereo", "temporal"),
        default="stereo",
        help="stereo: left/0 vs right/0; temporal: left/0 vs left/1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--coarse-res", type=int, default=560)
    parser.add_argument("--upsample-res", type=int, default=560)
    parser.add_argument("--num-matches", type=int, default=2000)
    return parser.parse_args()


def discover_pair(sequence_dir, pair_mode):
    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    left_images = sorted(
        path
        for path in (sequence_dir / "images" / "left").glob("*")
        if path.is_file() and path.suffix.lower() in image_suffixes
    )
    right_images = sorted(
        path
        for path in (sequence_dir / "images" / "right").glob("*")
        if path.is_file() and path.suffix.lower() in image_suffixes
    )
    if not left_images:
        raise FileNotFoundError(
            "No left images under {}/images/left".format(sequence_dir)
        )
    if pair_mode == "stereo":
        if not right_images:
            raise FileNotFoundError(
                "No right images under {}/images/right".format(sequence_dir)
            )
        return left_images[0], right_images[0]
    if len(left_images) < 2:
        raise FileNotFoundError("Temporal mode requires at least two left images")
    return left_images[0], left_images[1]


def main():
    args = parse_args()
    roma_root = args.roma_root.resolve()
    roma_vo = roma_root / "roma_vo.py"
    if not roma_vo.is_file():
        raise FileNotFoundError("Project roma_vo.py is missing: {}".format(roma_vo))

    sys.path.insert(0, str(roma_root))
    import torch
    from roma_vo import RoMaMatcher

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False; "
            "install the JetPack-matched PyTorch wheel first"
        )

    image_a, image_b = discover_pair(args.sequence_dir.resolve(), args.pair)
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    matcher = RoMaMatcher(
        device=args.device,
        coarse_res=args.coarse_res,
        upsample_res=args.upsample_res,
        use_custom_corr=False,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - start

    start = time.perf_counter()
    keypoints_a, keypoints_b, confidence = matcher.match(
        image_a,
        image_b,
        num_matches=args.num_matches,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    match_seconds = time.perf_counter() - start

    peak_memory_bytes = 0
    if args.device.startswith("cuda"):
        peak_memory_bytes = int(torch.cuda.max_memory_allocated())

    result = {
        "status": "OK",
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": args.device,
        "pair_mode": args.pair,
        "image_a": str(image_a),
        "image_b": str(image_b),
        "coarse_res": args.coarse_res,
        "upsample_res": args.upsample_res,
        "requested_matches": args.num_matches,
        "returned_matches": int(len(confidence)),
        "keypoints_a_shape": list(keypoints_a.shape),
        "keypoints_b_shape": list(keypoints_b.shape),
        "initialization_seconds": initialization_seconds,
        "match_seconds": match_seconds,
        "peak_cuda_memory_mib": peak_memory_bytes / (1024.0 * 1024.0),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
