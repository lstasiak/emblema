import pytest

from emblema.evaluation.domain.exceptions import InvalidTargetBinsError
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.target_bins import TargetBins
from tests.evaluation.support import window


def pool(*targets: float) -> list[LabelledWindow]:
    return [
        LabelledWindow(window=window("a", position, float(position)), target=target)
        for position, target in enumerate(targets)
    ]


def test_the_strata_hold_the_whole_pool_once() -> None:
    groups = TargetBins(3).of_pool(pool(5.0, 1.0, 9.0, 2.0, 7.0, 3.0))

    assert sorted(index for group in groups for index in group) == [0, 1, 2, 3, 4, 5]


def test_a_stratum_holds_the_windows_next_to_each_other_in_target() -> None:
    groups = TargetBins(3).of_pool(pool(5.0, 1.0, 9.0, 2.0, 7.0, 3.0))

    assert groups == ((1, 3), (5, 0), (4, 2))


def test_strata_come_out_within_one_window_of_each_other() -> None:
    groups = TargetBins(3).of_pool(pool(*(float(n) for n in range(10))))

    assert sorted(len(group) for group in groups) == [3, 3, 4]


def test_targets_tied_at_the_ceiling_are_spread_rather_than_piled_into_one_stratum() -> None:
    # A quarter of a real pool sits exactly at the ceiling; value thresholds would leave strata
    # empty, so the cut goes by rank.
    groups = TargetBins(4).of_pool(pool(125.0, 125.0, 125.0, 125.0, 10.0, 20.0, 30.0, 40.0))

    assert all(len(group) == 2 for group in groups)


def test_a_pool_too_small_to_fill_the_strata_is_refused() -> None:
    with pytest.raises(InvalidTargetBinsError, match="cannot fill"):
        TargetBins(4).of_pool(pool(1.0, 2.0, 3.0))


@pytest.mark.parametrize("count", [0, -1])
def test_a_stratification_without_bins_is_refused(count: int) -> None:
    with pytest.raises(InvalidTargetBinsError, match="at least one bin"):
        TargetBins(count)
