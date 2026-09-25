"""The ablation on a known answer: steady repeats add no share, noisy ones do."""

import pytest

from emblema.evaluation.adapters.synthetic.known_answer import KnownAnswer
from emblema.evaluation.domain.exceptions import (
    InvalidTwoLevelBootstrapError,
    InvalidUncertaintyDecompositionError,
)
from emblema.evaluation.domain.statistics.bootstrap_interval import BootstrapInterval
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.statistics.paired_unit_errors import PairedUnitErrors
from emblema.evaluation.domain.statistics.two_level_bootstrap import TwoLevelBootstrap
from emblema.evaluation.domain.statistics.uncertainty_decomposition import (
    UncertaintyDecomposition,
)

BOOTSTRAP = TwoLevelBootstrap(resamples=2000, seed=1)


def test_repeats_that_differ_by_nothing_add_no_share_and_no_width() -> None:
    steady = KnownAnswer(control_rmse=30.0, candidate_rmse=25.0, repeat_scatter=0.0)

    found = BOOTSTRAP.decompose(steady.repeats(5, seed=3))

    assert found.share_from_repeats == pytest.approx(0.0, abs=0.05)
    assert found.widening == pytest.approx(0.0, abs=0.3)


def test_noisy_repeats_add_a_share_and_widen_the_interval() -> None:
    noisy = KnownAnswer(
        control_rmse=30.0, candidate_rmse=25.0, unit_scatter=0.05, repeat_scatter=0.3
    )

    found = BOOTSTRAP.decompose(noisy.repeats(5, seed=3))

    assert found.share_from_repeats > 0.3
    assert found.widening > 0.0
    assert found.over_units_and_repeats.low <= found.over_units.low
    assert found.over_units.high <= found.over_units_and_repeats.high


def test_the_units_only_interval_is_the_registered_one_over_the_pooled_pair() -> None:
    repeats = KnownAnswer(control_rmse=30.0, candidate_rmse=25.0).repeats(3, seed=5)
    pooled = PairedUnitErrors.pooled([r.control for r in repeats], [r.candidate for r in repeats])

    found = BOOTSTRAP.decompose(repeats)

    assert found.over_units == PairedUnitBootstrap(resamples=2000, seed=1).compare(pooled).interval


def test_a_single_repeat_has_no_outer_level_to_resample_and_adds_nothing() -> None:
    found = BOOTSTRAP.decompose(
        KnownAnswer(control_rmse=30.0, candidate_rmse=25.0).repeats(1, seed=3)
    )

    assert found.share_from_repeats == pytest.approx(0.0, abs=0.05)


def test_the_same_seed_gives_the_same_decomposition() -> None:
    repeats = KnownAnswer(control_rmse=30.0, candidate_rmse=25.0).repeats(3, seed=5)

    assert BOOTSTRAP.decompose(repeats) == BOOTSTRAP.decompose(repeats)


def test_a_bootstrap_without_a_repeat_a_resample_or_a_level_is_refused() -> None:
    with pytest.raises(InvalidTwoLevelBootstrapError, match="needs a repeat"):
        BOOTSTRAP.decompose([])
    with pytest.raises(InvalidTwoLevelBootstrapError, match="resamples must be positive"):
        TwoLevelBootstrap(resamples=0)
    with pytest.raises(InvalidTwoLevelBootstrapError, match="level must lie in"):
        TwoLevelBootstrap(level=1.0)


def test_a_decomposition_at_two_levels_or_with_a_share_out_of_range_is_refused() -> None:
    narrow = BootstrapInterval(low=1.0, high=2.0, level=0.9)
    wide = BootstrapInterval(low=0.5, high=2.5, level=0.95)

    with pytest.raises(InvalidUncertaintyDecompositionError, match="one level"):
        UncertaintyDecomposition(
            over_units=narrow, over_units_and_repeats=wide, share_from_repeats=0.5
        )
    with pytest.raises(InvalidUncertaintyDecompositionError, match="must lie in"):
        UncertaintyDecomposition(
            over_units=narrow,
            over_units_and_repeats=BootstrapInterval(low=0.5, high=2.5, level=0.9),
            share_from_repeats=1.5,
        )
