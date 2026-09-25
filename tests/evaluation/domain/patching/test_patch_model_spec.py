from dataclasses import replace

import pytest

from emblema.evaluation.domain.exceptions import InvalidPatchModelSpecError
from tests.evaluation.support import patch_spec


@pytest.mark.parametrize(
    "field", ["patch_length", "stride", "width", "heads", "layers", "feedforward_width"]
)
def test_a_count_that_is_not_positive_is_refused(field: str) -> None:
    with pytest.raises(InvalidPatchModelSpecError, match=field):
        patch_spec(**{field: 0})


def test_a_stride_longer_than_the_patch_would_skip_steps_and_is_refused() -> None:
    with pytest.raises(InvalidPatchModelSpecError, match="skip steps"):
        patch_spec(patch_length=4, stride=5)


def test_a_width_the_heads_cannot_split_evenly_is_refused() -> None:
    with pytest.raises(InvalidPatchModelSpecError, match="heads"):
        patch_spec(width=10, heads=4)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, float("nan")])
def test_a_dropout_outside_the_unit_interval_is_refused(dropout: float) -> None:
    with pytest.raises(InvalidPatchModelSpecError, match="dropout"):
        patch_spec(dropout=dropout)


@pytest.mark.parametrize("resolution", [0.0, -1.0, float("inf"), float("nan")])
def test_a_resolution_that_is_not_positive_and_finite_is_refused(resolution: float) -> None:
    with pytest.raises(InvalidPatchModelSpecError, match="grid_resolution"):
        patch_spec(grid_resolution=resolution)


def test_a_window_is_laid_on_its_span_in_corpus_time_rounded_up() -> None:
    assert patch_spec().steps_over(50.0) == 50
    assert patch_spec(grid_resolution=0.5).steps_over(49.0) == 25


def test_the_padded_end_gives_the_last_reading_a_patch_of_its_own() -> None:
    # Fifty steps padded by four: patches start at 0, 4, ..., 44 — twelve of them.
    spec = patch_spec(patch_length=8, stride=4)

    assert spec.patches_over(50) == 12
    assert spec.patches_over(8) == 2


def test_a_row_shorter_than_one_patch_is_refused() -> None:
    with pytest.raises(InvalidPatchModelSpecError, match="shorter than a patch"):
        patch_spec(patch_length=8).patches_over(7)


def test_the_shape_is_recorded_field_by_field() -> None:
    spec = patch_spec()

    assert list(spec.parameters()) == [
        "patch_length",
        "stride",
        "width",
        "heads",
        "layers",
        "feedforward_width",
        "dropout",
        "grid_resolution",
    ]
    assert replace(spec, width=16).parameters()["width"] == 16
