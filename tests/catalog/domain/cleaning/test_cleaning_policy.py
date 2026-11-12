import pytest

from emblema.catalog.domain.cleaning.cleaning_policy import CleaningPolicy
from emblema.catalog.domain.cleaning.value_bounds import ValueBounds
from emblema.catalog.domain.exceptions import InvalidCleaningPolicyError
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.static_feature import StaticFeature


def bounds(channel: str, low: float, high: float) -> ValueBounds:
    return ValueBounds(
        channel=channel,
        removed_below=low,
        clipped_below=low,
        clipped_above=high,
        removed_above=high,
    )


POLICY = CleaningPolicy((bounds("pH", 6.3, 10.0), bounds("Height", 0.0, 240.0)))


def test_an_observation_outside_its_bounds_is_removed() -> None:
    assert POLICY.cleaned(Observation("pH", 2.0, 735.0)) is None


def test_an_observation_inside_its_bounds_keeps_its_channel_and_time() -> None:
    observation = Observation("pH", 2.0, 7.38)

    assert POLICY.cleaned(observation) == observation


def test_a_channel_the_policy_does_not_name_is_left_as_it_is() -> None:
    observation = Observation("HR", 1.0, 900.0)

    assert POLICY.cleaned(observation) == observation


def test_a_descriptor_is_held_to_its_bounds_as_an_observation_is() -> None:
    assert POLICY.cleaned(StaticFeature("Height", 300.0)) is None
    assert POLICY.cleaned(StaticFeature("Height", 170.0)) == StaticFeature("Height", 170.0)


def test_the_policy_names_its_channels() -> None:
    assert POLICY.channels == {"pH", "Height"}


def test_a_policy_naming_no_channel_is_refused() -> None:
    with pytest.raises(InvalidCleaningPolicyError, match="at least one"):
        CleaningPolicy(())


def test_a_channel_bounded_twice_is_refused() -> None:
    with pytest.raises(InvalidCleaningPolicyError, match=r"twice.*pH"):
        CleaningPolicy((bounds("pH", 6.3, 10.0), bounds("pH", 6.0, 9.0)))
