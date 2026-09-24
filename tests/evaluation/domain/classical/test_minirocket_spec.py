import pytest

from emblema.evaluation.domain.classical.minirocket_spec import MiniRocketSpec
from emblema.evaluation.domain.exceptions import InvalidMiniRocketSpecError


@pytest.mark.parametrize("features", [0, 83, 85, 10_000])
def test_a_count_that_is_not_a_whole_multiple_of_the_kernel_family_is_refused(
    features: int,
) -> None:
    with pytest.raises(InvalidMiniRocketSpecError, match="multiple of 84"):
        MiniRocketSpec(features=features)


def test_by_default_a_window_is_laid_on_one_step_per_unit_of_its_corpus_time() -> None:
    spec = MiniRocketSpec(features=84)

    assert spec.steps_over(50.0) == 50
    assert spec.parameters() == {"convolution_features": 84, "grid_resolution": 1.0}


def test_a_finer_grid_rounds_the_steps_up_so_no_part_of_the_window_is_dropped() -> None:
    assert MiniRocketSpec(features=84, grid_resolution=1.28).steps_over(50.0) == 64
    assert MiniRocketSpec(features=84, grid_resolution=0.5).steps_over(49.0) == 25


@pytest.mark.parametrize("resolution", [0.0, -1.0, float("inf"), float("nan")])
def test_a_resolution_that_is_not_positive_and_finite_is_refused(resolution: float) -> None:
    with pytest.raises(InvalidMiniRocketSpecError, match="grid_resolution"):
        MiniRocketSpec(features=84, grid_resolution=resolution)
