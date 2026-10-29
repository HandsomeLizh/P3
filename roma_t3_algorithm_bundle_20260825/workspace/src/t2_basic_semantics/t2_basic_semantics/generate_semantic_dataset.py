from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .segmenter import BasicSemanticSegmenter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(Path.home() / "RoMa" / "data" / "data1223"),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    left_dir = data_dir / "left"
    if not left_dir.is_dir():
        left_dir = data_dir / "images" / "left"
    if not left_dir.is_dir():
        raise FileNotFoundError(f"Left image directory does not exist: {left_dir}")
    label_dir = data_dir / "semantic_labels"
    color_dir = data_dir / "semantic_color"
    label_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)

    segmenter = BasicSemanticSegmenter()
    images = sorted(
        [
            path
            for path in left_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ],
        key=lambda path: int(path.stem),
    )

    for index, path in enumerate(images, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[warning] cannot read {path}")
            continue
        labels = segmenter.segment(image)
        color = segmenter.colorize(labels)
        cv2.imwrite(str(label_dir / path.name), labels)
        cv2.imwrite(str(color_dir / path.name), color)
        print(f"[{index}/{len(images)}] {path.name}")

    print(f"Labels: {label_dir}")
    print(f"Colours: {color_dir}")


if __name__ == "__main__":
    main()
