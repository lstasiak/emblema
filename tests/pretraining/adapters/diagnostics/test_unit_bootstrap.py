"""A kind's interval is a bootstrap over units, and its errors are ratios of sums over them."""

import pytest

from emblema.pretraining.adapters.diagnostics.unit_bootstrap import UnitBootstrap
from emblema.pretraining.domain.assessment.interval import Verdict
from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally
from emblema.pretraining.domain.mask_kind import MaskKind
from tests.support.reconstruction_runs import tallies

summarise = UnitBootstrap().summarise


def test_the_interval_lies_above_zero_only_when_the_units_agree() -> None:
    clear = summarise(tallies(MaskKind.TOKEN, model=0.02, matched=0.04, spread=0.005))[0]
    scattered = summarise(tallies(MaskKind.TOKEN, model=0.04, matched=0.04, spread=0.03))[0]
    worse = summarise(tallies(MaskKind.TOKEN, model=0.06, matched=0.04, spread=0.005))[0]

    assert clear.verdict is Verdict.LEARNT
    assert scattered.matched_excess.low < 0.0 < scattered.matched_excess.high
    assert scattered.verdict is Verdict.MATCHED
    assert worse.verdict is Verdict.BEATEN


def test_the_summary_is_a_ratio_of_sums_over_units() -> None:
    uneven = [
        MaskKindTally(MaskKind.BLOCK, False, "u/0", 10, 1.0, 2.0, 1.5, 10.0, 0.1),
        MaskKindTally(MaskKind.BLOCK, False, "u/1", 90, 1.0, 2.0, 1.5, 90.0, 0.9),
    ]

    summary = summarise(uneven)[0]

    assert (summary.tokens, summary.units) == (100, 2)
    assert summary.model_error == pytest.approx(0.02)
    assert summary.matched_error == pytest.approx(0.04)
    assert summary.linear_error == pytest.approx(0.03)
    assert summary.noise_floor == pytest.approx(0.01)
