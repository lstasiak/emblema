"""The control's verdict is code: it decides whether the generator has the structure it claims.

A verdict that could only ever read "as designed" would put a rubber stamp on every note quoting
it, so the thresholds are exercised from both sides here — and the measurement itself is run on
each of the four corpora, cut to a handful of units.
"""

import argparse

import numpy as np
import pytest

from emblema.shared.adapters.synthetic.layouts import (
    CONTROL_A,
    CONTROL_B,
    CONTROL_PROCESS,
    LAYOUTS,
)
from emblema.shared.adapters.synthetic.sensor_layout import SensorLayout
from scripts.synthetic_control_report import (
    COUPLED_EXCESS,
    NULL_EXCESS,
    LayoutRecovery,
    Report,
    explained,
    measure,
    recover,
    verdict_section,
)

UNITS = 6


def recovery(*, coupling: float = 1.0, own: float = 0.9, shuffled: float = 0.2) -> LayoutRecovery:
    """A recovery that came out as designed unless a test says otherwise about its verdict."""
    return LayoutRecovery(
        layout="control-a",
        coupling=coupling,
        fits=8,
        skipped=0,
        own=own,
        shuffled=shuffled,
        weakest_own=0.8,
    )


def test_a_design_that_says_everything_explains_everything() -> None:
    values = np.array([1.0, 2.0, 4.0, 8.0])
    design = np.column_stack([values, np.ones(4)])

    assert explained(design, values) == pytest.approx(1.0)


def test_a_design_that_says_nothing_explains_nothing() -> None:
    values = np.array([1.0, -1.0, 1.0, -1.0])

    assert explained(np.ones((4, 1)), values) == pytest.approx(0.0)


def test_a_channel_that_never_varied_has_nothing_to_explain() -> None:
    assert explained(np.ones((4, 1)), np.full(4, 3.0)) == 0.0


@pytest.mark.parametrize("layout", [CONTROL_A, CONTROL_B], ids=lambda layout: layout.name)
def test_a_coupled_corpus_is_explained_by_its_own_factors_and_not_by_another_unit_s(
    layout: SensorLayout,
) -> None:
    measured = recover(CONTROL_PROCESS, layout.with_dials(units=UNITS))

    assert measured.excess >= COUPLED_EXCESS
    assert measured.holds


@pytest.mark.parametrize("layout", [CONTROL_A, CONTROL_B], ids=lambda layout: layout.name)
def test_an_uncoupled_corpus_is_explained_no_better_by_its_own_factors(
    layout: SensorLayout,
) -> None:
    measured = recover(CONTROL_PROCESS, layout.with_dials(units=UNITS, coupling=0.0))

    assert abs(measured.excess) <= NULL_EXCESS
    assert measured.holds


def test_a_corpus_whose_channels_are_all_too_short_is_refused_rather_than_averaged() -> None:
    # Averaging no fits at all would hand the verdict a nan, which reads as a corpus that failed
    # rather than as a corpus that was never measured.
    with pytest.raises(SystemExit, match="no channel of control-a reached"):
        recover(CONTROL_PROCESS, CONTROL_A.with_dials(units=2, shortest_unit=1, longest_unit=1))


def test_a_corpus_of_one_unit_has_no_baseline_to_shuffle_against() -> None:
    # The other unit a channel is fitted against would be itself, leaving every corpus with an
    # excess of nothing and the coupled ones reported as broken.
    with pytest.raises(SystemExit, match="needs a second unit"):
        recover(CONTROL_PROCESS, CONTROL_A.with_dials(units=1))


def test_a_coupled_corpus_that_recovers_nothing_does_not_hold() -> None:
    assert not recovery(own=0.3, shuffled=0.25).holds


def test_a_null_corpus_that_recovers_something_does_not_hold() -> None:
    assert not recovery(coupling=0.0, own=0.9, shuffled=0.2).holds


def test_the_verdict_names_the_corpus_that_came_out_wrong() -> None:
    text = verdict_section(
        Report(process=CONTROL_PROCESS, units=UNITS, recoveries=[recovery(own=0.3)], figures=[])
    )

    assert "NOT AS DESIGNED" in text
    assert "does not behave as specified" in text


def test_the_verdict_passes_when_every_corpus_came_out_as_designed() -> None:
    text = verdict_section(
        Report(process=CONTROL_PROCESS, units=UNITS, recoveries=[recovery()], figures=[])
    )

    assert "NOT AS DESIGNED" not in text
    assert "A pipeline that finds nothing here is at fault." in text


def test_the_report_measures_every_corpus_asked_for_and_draws_none_when_told_not_to() -> None:
    arguments = argparse.Namespace(layout=sorted(LAYOUTS), units=UNITS, out="unused", figures=False)

    report = measure(arguments)

    assert [measured.layout for measured in report.recoveries] == sorted(LAYOUTS)
    assert all(measured.holds for measured in report.recoveries)
    assert report.figures == []
