from types import SimpleNamespace

import pytest

from t3_upstream_adapters.field_path import (
    FieldPathError,
    optional_field,
    resolve_field,
)


def test_resolves_nested_field_and_array_index():
    message = SimpleNamespace(
        cameras=[
            SimpleNamespace(image="left"),
            SimpleNamespace(image="right"),
        ]
    )

    assert resolve_field(message, "cameras[0].image") == "left"
    assert resolve_field(message, "cameras[1].image") == "right"


def test_empty_optional_path_disables_one_output():
    message = SimpleNamespace(value=3)

    assert optional_field(message, "") is None
    assert optional_field(message, "value") == 3


@pytest.mark.parametrize(
    "path",
    ["", "cameras[-1]", "cameras[abc]", "cameras[0] image"],
)
def test_rejects_invalid_paths(path):
    with pytest.raises(FieldPathError):
        resolve_field(SimpleNamespace(cameras=[]), path)


def test_reports_missing_field_without_evaluating_expression():
    message = SimpleNamespace(stereo=SimpleNamespace(left="image"))

    with pytest.raises(FieldPathError, match="has no field 'right'"):
        resolve_field(message, "stereo.right")


def test_reports_out_of_range_index():
    message = SimpleNamespace(cameras=[])

    with pytest.raises(FieldPathError, match="cannot select index 2"):
        resolve_field(message, "cameras[2]")
