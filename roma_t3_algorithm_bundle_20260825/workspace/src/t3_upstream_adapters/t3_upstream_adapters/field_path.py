"""Strict dotted-field lookup used by the upstream-message templates."""

from __future__ import annotations

import re
from typing import Any, List, Optional


_SEGMENT_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?P<indices>(?:\[[0-9]+\])*)$"
)
_INDEX_PATTERN = re.compile(r"\[([0-9]+)\]")


class FieldPathError(ValueError):
    """Raised when an interface field path is invalid or cannot be resolved."""


def resolve_field(message: object, path: str) -> Any:
    """Resolve ``field.subfield[0]`` without evaluating arbitrary code."""

    normalised = str(path).strip()
    if not normalised:
        raise FieldPathError("field path cannot be empty")

    current: Any = message
    traversed: List[str] = []
    for segment in normalised.split("."):
        match = _SEGMENT_PATTERN.fullmatch(segment)
        if match is None:
            raise FieldPathError(
                f"invalid field-path segment {segment!r} in {normalised!r}"
            )
        field_name = match.group("name")
        traversed.append(field_name)
        if not hasattr(current, field_name):
            prefix = ".".join(traversed[:-1]) or "<root>"
            raise FieldPathError(
                f"{prefix} has no field {field_name!r}; "
                f"configured path={normalised!r}"
            )
        current = getattr(current, field_name)
        for index_text in _INDEX_PATTERN.findall(match.group("indices")):
            index = int(index_text)
            try:
                current = current[index]
            except (IndexError, KeyError, TypeError) as exception:
                raise FieldPathError(
                    f"cannot select index {index} in {normalised!r}"
                ) from exception
            traversed[-1] += f"[{index}]"
    return current


def optional_field(message: object, path: str) -> Optional[Any]:
    """Resolve a configured field, or return ``None`` for an empty path."""

    if not str(path).strip():
        return None
    return resolve_field(message, path)
