"""The procedure on data with a known answer: a shift is found, no shift is not, and how often."""

import pytest

from emblema.evaluation.adapters.synthetic.known_answer import KnownAnswer
from emblema.evaluation.domain.exceptions import InvalidPairedUnitBootstrapError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.scoring.unit_error import UnitError
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.statistics.paired_unit_errors import PairedUnitErrors

# Enough datasets to read a rate off, few enough resamples to keep the whole check in a second.
DATASETS = 200
QUICK = PairedUnitBootstrap(resamples=300, seed=1)


def test_a_true_reduction_is_found_with_its_whole_interval_above_zero() -> None:
    known = KnownAnswer(control_rmse=30.0, candidate_rmse=22.0)

    compared = PairedUnitBootstrap(resamples=2000, seed=1).compare(known.paired(seed=3))

    assert compared.reduction == pytest.approx(known.true_reduction, abs=1.5)
    assert compared.relative_reduction == pytest.approx(8.0 / 30.0, abs=0.05)
    assert compared.interval.above_zero
    assert compared.interval.low < compared.reduction < compared.interval.high
    assert compared.p_value < 0.01
    assert compared.confirms(0.10)
    assert compared.distinguishable


def test_no_true_difference_leaves_zero_inside_the_interval() -> None:
    known = KnownAnswer(control_rmse=30.0, candidate_rmse=30.0)

    compared = PairedUnitBootstrap(resamples=2000, seed=1).compare(known.paired(seed=5))

    assert abs(compared.reduction) < 1.0
    assert not compared.interval.excludes_zero
    assert compared.p_value > 0.2
    assert not compared.confirms(0.10)
    assert not compared.distinguishable


def test_over_many_datasets_the_interval_covers_the_true_reduction_about_as_often_as_stated() -> (
    None
):
    known = KnownAnswer(control_rmse=30.0, candidate_rmse=25.0)

    intervals = [QUICK.compare(known.paired(seed=seed)).interval for seed in range(DATASETS)]
    covered = sum(1 for found in intervals if found.low <= known.true_reduction <= found.high)

    # A percentile interval over eighteen units runs a little short of its nominal coverage,
    # which is a known property of the method and not a bug in it: the band allows for that and
    # would still catch an interval that was too narrow by half.
    assert 0.85 <= covered / DATASETS <= 0.99


def test_over_many_datasets_with_no_true_difference_zero_is_seldom_excluded() -> None:
    known = KnownAnswer(control_rmse=30.0, candidate_rmse=30.0)

    excluded = sum(
        1 for seed in range(DATASETS) if QUICK.compare(known.paired(seed=seed)).distinguishable
    )

    # The percentile interval over eighteen units excludes a true zero about twice as often as
    # its level says (measured in `docs/verification/verdict-statistics.md`); the band holds the
    # procedure to that known shortfall and would catch one that grew.
    assert 0.03 <= excluded / DATASETS <= 0.16


def test_the_same_seed_gives_the_same_interval_and_another_seed_a_near_one() -> None:
    compared = KnownAnswer(control_rmse=30.0, candidate_rmse=25.0).paired(seed=7)
    first = PairedUnitBootstrap(resamples=1000, seed=1).compare(compared)
    again = PairedUnitBootstrap(resamples=1000, seed=1).compare(compared)
    other = PairedUnitBootstrap(resamples=1000, seed=2).compare(compared)

    assert first == again
    assert first.interval.low == pytest.approx(other.interval.low, abs=0.3)
    assert first.interval.high == pytest.approx(other.interval.high, abs=0.3)


def test_a_wider_level_gives_a_wider_interval() -> None:
    compared = KnownAnswer(control_rmse=30.0, candidate_rmse=25.0).paired(seed=7)
    narrow = PairedUnitBootstrap(resamples=1000, seed=1, level=0.8).compare(compared).interval
    wide = PairedUnitBootstrap(resamples=1000, seed=1, level=0.99).compare(compared).interval

    assert wide.low < narrow.low
    assert narrow.high < wide.high
    assert (narrow.level, wide.level) == (0.8, 0.99)


def test_one_unit_resamples_to_itself_and_has_no_width() -> None:
    compared = PairedUnitErrors(
        control=(UnitError(unit=UnitKey("a"), squared_error=16.0, windows=1),),
        candidate=(UnitError(unit=UnitKey("a"), squared_error=4.0, windows=1),),
    )

    found = PairedUnitBootstrap(resamples=50, seed=1).compare(compared)

    assert (found.interval.low, found.interval.high) == (2.0, 2.0)
    # Every resample lies on one side, yet the p-value is not zero: the observed reduction
    # counts as one more, so fifty resamples can say no less than two in fifty-one.
    assert found.p_value == pytest.approx(2.0 / 51.0)


def test_the_resampled_reductions_come_back_in_order_and_the_interval_is_read_off_them() -> None:
    compared = KnownAnswer(control_rmse=30.0, candidate_rmse=25.0).paired(seed=7)
    bootstrap = PairedUnitBootstrap(resamples=500, seed=1)

    reductions = bootstrap.reductions(compared)

    assert reductions == sorted(reductions)
    assert len(reductions) == 500
    assert bootstrap.interval_of(reductions) == bootstrap.compare(compared).interval


def test_a_bootstrap_without_a_resample_is_refused() -> None:
    with pytest.raises(InvalidPairedUnitBootstrapError, match="resamples must be positive"):
        PairedUnitBootstrap(resamples=0)


@pytest.mark.parametrize("level", [0.0, 1.0])
def test_a_bootstrap_at_no_level_is_refused(level: float) -> None:
    with pytest.raises(InvalidPairedUnitBootstrapError, match="level must lie in"):
        PairedUnitBootstrap(level=level)
