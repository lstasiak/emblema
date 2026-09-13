"""The sanity report says whether a window still holds what it was cut from; that verdict is code.

The figures are for a person to look at, but the sentence beside them is arithmetic, and a
verdict that could only ever read "holds" would put a rubber stamp on every note that quotes it.
The report runs here on the miniature sample, over one window, into a temporary directory.
"""

import argparse
from pathlib import Path

import pytest

from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.catalog.domain.tokenisation.placed_window import PlacedWindow
from emblema.catalog.domain.tokenisation.window_reconstruction import WindowReconstruction
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from scripts.window_sanity_report import (
    SAMPLES,
    TOLERANCE,
    Report,
    WindowCheck,
    chosen_unit,
    chosen_window,
    corpus_root,
    default_window,
    measure,
    residual,
    spread_over,
    verdict_section,
)
from tests.shared.adapters.windows.support import TIMED
from tests.support.corpora import CORPUS

UNITS = [
    CorpusUnit(UnitKey("a"), TimeExtent(1.0, 10.0)),
    CorpusUnit(UnitKey("b"), TimeExtent(1.0, 100.0)),
]
PLACED = [
    PlacedWindow(UnitKey("a"), TimeExtent(float(start), float(start) + 1.0), TIMED)
    for start in range(10)
]


def arguments(out: Path, **overrides: object) -> argparse.Namespace:
    settings: dict[str, object] = {
        "corpus": CORPUS,
        "subset": "FD001",
        "unit": None,
        "windows": 1,
        "length": None,
        "stride": None,
        "out": str(out),
    }
    settings.update(overrides)
    return argparse.Namespace(**settings)


def test_a_corpus_that_was_never_fetched_falls_back_to_the_sample() -> None:
    root, source = corpus_root("never-downloaded")

    assert root == SAMPLES / "never-downloaded"
    assert source == "miniature sample"


def test_half_a_window_is_refused() -> None:
    with pytest.raises(SystemExit, match="both or neither"):
        chosen_window(arguments(Path("out"), length=20.0), CORPUS)


def test_a_window_given_beats_the_corpus_default() -> None:
    given = chosen_window(arguments(Path("out"), length=20.0, stride=7.0), CORPUS)

    assert given == WindowSpec(20.0, 7.0)
    assert given != default_window(CORPUS)


def test_the_unit_drawn_is_the_longest_one_where_a_drifting_axis_shows_most() -> None:
    assert chosen_unit(UNITS, None) is UNITS[1]


def test_a_unit_the_corpus_does_not_hold_stops_the_report() -> None:
    with pytest.raises(SystemExit, match="no unit 'zz'"):
        chosen_unit(UNITS, "zz")


def test_the_windows_drawn_are_spread_along_the_unit() -> None:
    # Three windows off one end of an engine's life would say nothing about the rest of it.
    assert spread_over(PLACED, 3) == [PLACED[0], PLACED[4], PLACED[9]]
    assert spread_over(PLACED, 20) == PLACED


def test_a_window_that_read_back_the_wrong_number_of_rows_stops_the_report() -> None:
    inside = [Observation("T2", 1.0, 0.5), Observation("T2", 2.0, 0.5)]

    with pytest.raises(SystemExit, match="read back 1 observations where 2"):
        residual(inside, (), WindowReconstruction((inside[0],), ()))


def test_a_window_that_dropped_a_static_feature_stops_the_report() -> None:
    with pytest.raises(SystemExit, match="read back 0 static features where 1"):
        residual((), (StaticFeature("age", 61.0),), WindowReconstruction((), ()))


def test_a_static_feature_that_came_back_changed_counts_as_a_value_error() -> None:
    feature = StaticFeature("age", 61.0)

    values, times = residual(
        (), (feature,), WindowReconstruction((), (StaticFeature("age", 62.0),))
    )

    assert (values, times) == (1.0, 0.0)


def test_an_observation_that_came_back_under_another_channel_stops_the_report() -> None:
    inside = [Observation("T2", 1.0, 0.5)]

    with pytest.raises(SystemExit, match="observations of other channels: T2, T3"):
        residual(inside, (), WindowReconstruction((Observation("T3", 1.0, 0.5),), ()))


def test_a_static_feature_that_came_back_under_another_channel_stops_the_report() -> None:
    # Sorting pairs the two sides by channel, so a rename that kept the value would otherwise be
    # subtracted from itself and reported as a round trip that held.
    with pytest.raises(SystemExit, match="static features of other channels: age, hours"):
        residual(
            (),
            (StaticFeature("age", 61.0),),
            WindowReconstruction((), (StaticFeature("hours", 61.0),)),
        )


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> Report:
    """One run of the report over the sample; drawing the figure is the slow part of it."""
    return measure(arguments(tmp_path_factory.mktemp("figures")))


def test_the_sample_round_trips_and_the_verdict_says_so(report: Report) -> None:
    assert len(report.checks) == 1
    assert max(report.checks[0].value_residual, report.checks[0].time_residual) <= TOLERANCE
    assert "The round trip holds" in verdict_section(report)
    assert report.checks[0].figure.is_file()


def test_a_round_trip_beyond_the_tolerance_is_named_in_the_verdict(report: Report) -> None:
    # The verdict is the only sentence a reader of the note takes on trust, so it has to be able
    # to say no; an assertion on the happy case alone would pass on a verdict wired shut.
    strayed = _with_residual(report, TOLERANCE * 10)

    assert "The round trip DOES NOT HOLD" in verdict_section(strayed)


def _with_residual(report: Report, value: float) -> Report:
    check = report.checks[0]
    strayed = WindowCheck(
        check.unit, check.extent, check.tokens, value, check.time_residual, check.figure
    )
    return Report(
        corpus=report.corpus,
        source=report.source,
        root=report.root,
        window=report.window,
        units=report.units,
        short_units=report.short_units,
        channels=report.channels,
        checks=[strayed],
    )
