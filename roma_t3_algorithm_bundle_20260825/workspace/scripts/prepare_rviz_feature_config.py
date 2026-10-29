#!/usr/bin/env python3
"""Enable one named RViz display in a temporary copy of the base config."""

from __future__ import annotations

import argparse
from pathlib import Path


DISPLAY_NAME = "Stereo and Temporal Feature Matches"


def configure(source: Path, destination: Path, *, enabled: bool) -> None:
    lines = Path(source).read_text(encoding="utf-8").splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if line.startswith("    - ")]
    starts.append(len(lines))
    matching_ranges = []
    marker = f"      Name: {DISPLAY_NAME}"
    for begin, end in zip(starts[:-1], starts[1:]):
        if any(line.rstrip("\r\n") == marker for line in lines[begin:end]):
            matching_ranges.append((begin, end))
    if len(matching_ranges) != 1:
        raise ValueError(
            f"Expected one RViz display named {DISPLAY_NAME!r}, "
            f"found {len(matching_ranges)}"
        )

    begin, end = matching_ranges[0]
    state = "true" if enabled else "false"
    enabled_updates = 0
    value_updates = 0
    for index in range(begin, end):
        stripped = lines[index].rstrip("\r\n")
        newline = "\r\n" if lines[index].endswith("\r\n") else "\n"
        if stripped.startswith("      Enabled:"):
            lines[index] = f"      Enabled: {state}{newline}"
            enabled_updates += 1
        elif stripped.startswith("      Value:"):
            lines[index] = f"      Value: {state}{newline}"
            value_updates += 1
    if enabled_updates != 1 or value_updates != 1:
        raise ValueError(
            "Feature-match RViz display must contain one Enabled and one "
            "display-level Value field"
        )
    Path(destination).write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("state", choices=("enabled", "disabled"))
    args = parser.parse_args()
    configure(
        args.source,
        args.destination,
        enabled=args.state == "enabled",
    )


if __name__ == "__main__":
    main()
