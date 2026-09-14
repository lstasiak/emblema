"""Contract of the MaskKindSummariser port, run against every adapter.

What every adapter owes, whatever statistics it resamples with: one row per kind and side, errors
that are ratios of sums over the groups, an interval that cannot hold zero when every group agrees
on the sign, and the same answer twice for the same tallies — two runs of a configuration are
compared through this port, so a verdict that moved must have moved in the run.

One adapter exists. The list is a list so that a second one is held to the same tests by being
added to it, and so that the obligations are written down where the port is, not inside the only
adapter that meets them today.
"""

from collections.abc import Callable

import pytest

from emblema.pretraining.adapters.diagnostics.unit_bootstrap import UnitBootstrap
from emblema.pretraining.domain.assessment.interval import Verdict
from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.ports.mask_kind_summariser import MaskKindSummariser
from tests.support.reconstruction_runs import tallies

ADAPTERS: dict[str, Callable[[], MaskKindSummariser]] = {"unit_bootstrap": UnitBootstrap}


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def summariser(request: pytest.FixtureRequest) -> MaskKindSummariser:
    factory: Callable[[], MaskKindSummariser] = request.param
    return factory()


def test_nothing_tallied_is_summarised_as_nothing(summariser: MaskKindSummariser) -> None:
    assert summariser.summarise([]) == ()


def test_each_kind_and_side_gets_one_row_with_the_channels_apart_last(
    summariser: MaskKindSummariser,
) -> None:
    mixed = [
        *tallies(MaskKind.TOKEN, model=0.02, matched=0.04),
        *tallies(MaskKind.CHANNEL, model=0.10, matched=0.30),
        *tallies(MaskKind.CHANNEL, model=0.90, matched=0.95, apart=True),
    ]

    summaries = summariser.summarise(mixed)

    assert [(row.kind, row.apart) for row in summaries] == [
        (MaskKind.CHANNEL, False),
        (MaskKind.TOKEN, False),
        (MaskKind.CHANNEL, True),
    ]


def test_an_error_is_the_ratio_of_the_summed_errors_to_the_summed_tokens(
    summariser: MaskKindSummariser,
) -> None:
    uneven = [
        MaskKindTally(MaskKind.BLOCK, False, "u/0", 10, 1.0, 2.0, 1.5, 10.0, 0.1),
        MaskKindTally(MaskKind.BLOCK, False, "u/1", 90, 1.0, 2.0, 1.5, 90.0, 0.9),
    ]

    summary = summariser.summarise(uneven)[0]

    assert (summary.tokens, summary.units) == (100, 2)
    assert summary.model_error == pytest.approx(0.02)
    assert summary.matched_error == pytest.approx(0.04)
    assert summary.linear_error == pytest.approx(0.03)
    assert summary.mean_error == pytest.approx(1.0)
    assert summary.noise_floor == pytest.approx(0.01)


def test_a_floor_missing_from_one_group_leaves_the_kind_without_one(
    summariser: MaskKindSummariser,
) -> None:
    partial = [
        MaskKindTally(MaskKind.BLOCK, False, "u/0", 10, 1.0, 2.0, 1.5, 10.0, 0.1),
        MaskKindTally(MaskKind.BLOCK, False, "u/1", 10, 1.0, 2.0, 1.5, 10.0, None),
    ]

    assert summariser.summarise(partial)[0].noise_floor is None


@pytest.mark.parametrize(
    ("model", "matched", "verdict"),
    [(0.02, 0.04, Verdict.LEARNT), (0.06, 0.04, Verdict.BEATEN)],
)
def test_groups_that_all_agree_on_the_sign_leave_no_room_for_zero(
    summariser: MaskKindSummariser, model: float, matched: float, verdict: Verdict
) -> None:
    # Every group has the same sign of excess, so no resampling of groups can produce a sum of the
    # other sign: an interval holding zero here would be the statistics talking, not the numbers.
    agreed = tallies(MaskKind.TOKEN, model=model, matched=matched, spread=0.001)

    summary = summariser.summarise(agreed)[0]

    assert summary.verdict is verdict
    assert (summary.matched_excess.low > 0.0) is (verdict is Verdict.LEARNT)


def test_groups_that_disagree_leave_zero_inside_the_interval(
    summariser: MaskKindSummariser,
) -> None:
    scattered = tallies(MaskKind.TOKEN, model=0.04, matched=0.04, spread=0.03)

    summary = summariser.summarise(scattered)[0]

    assert summary.matched_excess.low < 0.0 < summary.matched_excess.high
    assert summary.verdict is Verdict.MATCHED


def test_the_same_tallies_are_summarised_the_same_way_twice(
    summariser: MaskKindSummariser,
) -> None:
    given = tallies(MaskKind.BLOCK, model=0.05, matched=0.25, linear=0.15, spread=0.02)

    assert summariser.summarise(given) == summariser.summarise(given)


def test_both_baselines_are_measured_against_the_same_model_error(
    summariser: MaskKindSummariser,
) -> None:
    both = tallies(MaskKind.BLOCK, model=0.05, matched=0.25, linear=0.15, spread=0.01)

    summary = summariser.summarise(both)[0]

    assert summary.matched_excess.low > summary.linear_excess.low
    assert summary.excess == pytest.approx(summary.matched_error - summary.model_error)
