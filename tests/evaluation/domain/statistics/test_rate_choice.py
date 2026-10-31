import pytest

from emblema.evaluation.domain.exceptions import InvalidRateChoiceError
from emblema.evaluation.domain.statistics.rate_choice import RateChoice, RateOption


def choice(*options: tuple[float, float, float]) -> RateChoice:
    return RateChoice(
        in_force=0.001,
        others=tuple(RateOption(rate=r, gain=g, standard_error=e) for r, g, e in options),
    )


def test_the_rate_in_force_stays_where_no_other_clears_its_error() -> None:
    kept = choice((0.0001, -0.01, 0.005), (0.003, 0.004, 0.005))

    assert kept.chosen == 0.001
    assert not kept.at_edge


def test_a_gain_of_exactly_one_error_does_not_clear() -> None:
    assert choice((0.003, 0.005, 0.005)).chosen == 0.001


def test_of_several_rates_that_clear_the_largest_gain_is_kept_and_an_edge_named() -> None:
    kept = choice((0.0001, 0.02, 0.005), (0.000333, 0.03, 0.005), (0.003, 0.01, 0.005))

    assert kept.chosen == 0.000333
    assert not kept.at_edge
    assert choice((0.003, 0.02, 0.005), (0.0001, 0.0, 0.005)).at_edge


@pytest.mark.parametrize(
    "options",
    [
        ((0.001, 0.0, 0.0),),
        ((0.003, float("nan"), 0.0),),
        ((0.003, 0.0, -0.1),),
        ((0.0, 0.0, 0.0),),
    ],
)
def test_a_grid_that_repeats_a_rate_or_holds_a_bad_figure_is_refused(
    options: tuple[tuple[float, float, float], ...],
) -> None:
    with pytest.raises(InvalidRateChoiceError):
        choice(*options)


def test_a_rate_in_force_out_of_range_is_refused() -> None:
    with pytest.raises(InvalidRateChoiceError, match="in force"):
        RateChoice(in_force=0.0, others=())


def test_a_setting_at_the_rate_in_force_is_read_by_the_same_rule_and_lies_at_no_edge() -> None:
    lpft = RateOption(
        rate=0.001, gain=0.02, standard_error=0.005, setting="encoder_rate=0.1,head_start=solved"
    )
    kept = RateChoice(
        in_force=0.001,
        others=(RateOption(rate=0.0001, gain=0.01, standard_error=0.005), lpft),
    )

    assert kept.chosen_option is lpft
    assert kept.chosen == 0.001
    assert not kept.at_edge


def test_a_setting_named_twice_is_refused_and_the_in_force_rate_stays_without_one() -> None:
    twice = RateOption(rate=0.001, gain=0.0, standard_error=0.0, setting="encoder_rate=0.1")
    with pytest.raises(InvalidRateChoiceError, match="twice"):
        RateChoice(in_force=0.001, others=(twice, twice))
    assert choice((0.0001, -0.01, 0.005)).chosen_option is None
