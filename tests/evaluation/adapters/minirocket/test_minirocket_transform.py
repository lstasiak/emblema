import numpy as np
import pytest

from emblema.evaluation.adapters.minirocket.minirocket_transform import MiniRocketTransform

SERIES = np.random.default_rng(0).normal(size=(12, 4, 40))


def fitted(features: int = 840, seed: int = 1) -> MiniRocketTransform:
    return MiniRocketTransform.fitted(SERIES, features, np.random.default_rng(seed))


def test_every_feature_is_a_share_of_positions_and_there_are_as_many_as_asked() -> None:
    rows = fitted().of(SERIES)

    assert rows.shape == (12, 840)
    assert ((rows >= 0.0) & (rows <= 1.0)).all()


def test_a_count_between_multiples_of_the_family_is_rounded_down() -> None:
    assert fitted(features=900).width == 840


def test_the_same_seed_fits_the_same_transform_and_another_does_not() -> None:
    first, again, other = fitted().of(SERIES), fitted().of(SERIES), fitted(seed=2).of(SERIES)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_the_dilations_reach_as_far_as_the_series_allows_and_no_further() -> None:
    transform = fitted()

    assert transform.dilations[0] == 1
    assert 8 * transform.dilations[-1] <= SERIES.shape[2] - 1


def test_the_biases_split_the_series_they_were_drawn_from() -> None:
    # Biases are quantiles of a convolution of the fitted series, so on those series the share
    # above them is neither nothing nor everything.
    assert 0.3 < fitted().of(SERIES).mean() < 0.7


def test_a_combination_never_names_a_channel_twice_or_one_that_does_not_exist() -> None:
    transform = fitted()
    start = 0
    for size in transform.combination_sizes.tolist():
        chosen = transform.channels[start : start + size]
        assert len(set(chosen.tolist())) == size
        assert chosen.max() < SERIES.shape[1]
        start += size


def test_series_of_another_length_are_refused() -> None:
    with pytest.raises(ValueError, match="fitted to 40"):
        fitted().of(SERIES[:, :, :30])


@pytest.mark.parametrize(("steps", "features"), [(8, 84), (20, 83)])
def test_too_short_a_series_or_too_few_features_is_refused(steps: int, features: int) -> None:
    with pytest.raises(ValueError, match="needs 9 steps and 84 features"):
        MiniRocketTransform.fitted(SERIES[:, :, :steps], features, np.random.default_rng(1))
