"""A pretext's error read as a detector on the fitting side's later months: the readings' shape."""

from pathlib import Path

import pytest

from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from scripts.history_pretext_reading_pilot import (
    INTERVALS,
    READINGS,
    Interval,
    Reading,
    file_name_of,
    interval_table,
    intervals_of,
    read,
    read_predictions,
    reading_table,
    readings_of,
    write_predictions,
    write_readings,
)
from tests.evaluation.support import window


def prediction(month: str, position: int, target: float, predicted: float) -> WindowPrediction:
    return WindowPrediction(
        window=window(f"ESA-Mission1/{month}", position, float(position)),
        target=target,
        predicted=predicted,
    )


# Two months: an event found at the top in the first, a normal window scored highest in the
# second, which the pretraining never read.
PREDICTIONS = (
    prediction("2004-01", 0, 0.0, 0.1),
    prediction("2004-01", 1, 7.0, 0.9),
    prediction("2004-01", 2, 0.0, 0.2),
    prediction("2006-10", 3, 0.0, 0.8),
    prediction("2006-10", 4, 9.0, 0.5),
)

# Three months, each with an event and a normal window, so every resample of months can be read.
ARM = (
    prediction("2004-01", 0, 0.0, 0.1),
    prediction("2004-01", 1, 7.0, 0.9),
    prediction("2004-02", 2, 0.0, 0.3),
    prediction("2004-02", 3, 8.0, 0.7),
    prediction("2004-03", 4, 0.0, 0.6),
    prediction("2004-03", 5, 9.0, 0.4),
)
# The other candidate ranks the third month's event above its normal window and the first's below.
OTHER = (
    prediction("2004-01", 0, 0.0, 0.9),
    prediction("2004-01", 1, 7.0, 0.1),
    prediction("2004-02", 2, 0.0, 0.3),
    prediction("2004-02", 3, 8.0, 0.7),
    prediction("2004-03", 4, 0.0, 0.4),
    prediction("2004-03", 5, 9.0, 0.6),
)


def test_a_candidate_is_read_over_every_month_and_over_the_months_the_pretraining_never_read() -> (
    None
):
    readings = readings_of(
        "pretext_error@error=largest", PREDICTIONS, unread_from="2006-10", share=0.5, seconds=3.0
    )

    assert [(r.months, r.windows, r.events) for r in readings] == [("all", 5, 2), ("unread", 2, 1)]
    # Over every month the first event is found at precision one and the second at two thirds,
    # one normal window scoring above it; over the unread months alone, at a half.
    assert readings[0].area == pytest.approx(5 / 6)
    assert readings[1].area == pytest.approx(0.5)
    assert readings[0].seconds == readings[1].seconds == 3.0


def test_months_without_both_an_event_and_a_normal_window_are_not_read_apart() -> None:
    readings = readings_of(
        "channel_local", PREDICTIONS[:3], unread_from="2006-10", share=0.5, seconds=1.0
    )

    assert [r.months for r in readings] == ["all"]


def test_the_table_and_the_file_names_carry_the_candidate_s_knobs() -> None:
    reading = Reading(
        candidate="pretext_error@error=largest,hidden=tail",
        months="all",
        windows=5,
        events=2,
        area=0.75,
        event_recall=1.0,
        seconds=3.0,
    )

    assert "| pretext_error@error=largest,hidden=tail | all | 5 | 2 | 0.750 | 1.000 | 3 |" in (
        reading_table([reading])
    )
    assert file_name_of(reading.candidate) == "pretext_error--error-largest+hidden-tail"


def test_an_interval_brackets_each_reading_and_the_arm_s_lead_over_every_other_candidate() -> None:
    rankings = {"arm": AnomalyRanking.of(ARM), "other": AnomalyRanking.of(OTHER)}

    intervals = intervals_of(rankings, resamples=200, seed=1, share=0.5)

    assert [(i.reading, i.measure) for i in intervals] == [
        ("arm", "area"),
        ("arm", "event_recall"),
        ("other", "area"),
        ("other", "event_recall"),
        ("arm - other", "area"),
        ("arm - other", "event_recall"),
    ]
    by_reading = {(i.reading, i.measure): i for i in intervals}
    assert by_reading["arm", "area"].point == pytest.approx(rankings["arm"].average_precision)
    assert by_reading["arm - other", "area"].point == pytest.approx(
        rankings["arm"].average_precision - rankings["other"].average_precision
    )
    for interval in intervals:
        assert interval.low <= interval.point <= interval.high
    # The arm's lead is read on the same resamples as the two readings it is the difference of.
    assert by_reading["arm - other", "area"].high <= (
        by_reading["arm", "area"].high - by_reading["other", "area"].low + 1e-9
    )


def test_a_candidate_that_scored_other_months_than_the_arm_is_refused() -> None:
    rankings = {"arm": AnomalyRanking.of(ARM), "short": AnomalyRanking.of(OTHER[:4])}

    with pytest.raises(ValueError, match="other months"):
        intervals_of(rankings, resamples=10, seed=1, share=0.5)


def test_read_puts_the_intervals_beside_a_score_run_s_files_in_the_order_it_scored(
    tmp_path: Path,
) -> None:
    write_predictions(tmp_path / f"{file_name_of('arm@k=1')}.csv", "arm@k=1", ARM)
    write_predictions(tmp_path / "other.csv", "other", OTHER)
    write_readings(
        tmp_path / READINGS,
        [
            Reading(
                candidate=name,
                months="all",
                windows=6,
                events=3,
                area=0.5,
                event_recall=1.0,
                seconds=1.0,
            )
            for name in ("arm@k=1", "other")
        ],
    )

    intervals = read(tmp_path, resamples=50, seed=1, share=0.5)

    assert [i.reading for i in intervals][::2] == ["arm@k=1", "other", "arm@k=1 - other"]
    assert (tmp_path / INTERVALS).read_text().splitlines()[0] == "reading,measure,point,low,high"
    assert "| arm@k=1 - other | area |" in interval_table(intervals)


def test_predictions_written_are_read_back_whole(tmp_path: Path) -> None:
    write_predictions(tmp_path / "arm.csv", "arm", ARM)

    candidate, predictions = read_predictions(tmp_path / "arm.csv")

    assert candidate == "arm"
    assert predictions == list(ARM)


def test_a_predictions_file_of_two_candidates_or_none_is_refused(tmp_path: Path) -> None:
    (tmp_path / "mixed.csv").write_text(
        "candidate,unit,position,ends_at,target,predicted\n"
        "a,ESA-Mission1/2004-01,0,0.0,0.0,0.1\n"
        "b,ESA-Mission1/2004-01,1,1.0,7.0,0.9\n"
    )
    (tmp_path / "empty.csv").write_text("candidate,unit,position,ends_at,target,predicted\n")

    with pytest.raises(ValueError, match="2 candidates"):
        read_predictions(tmp_path / "mixed.csv")
    with pytest.raises(ValueError, match="0 candidates"):
        read_predictions(tmp_path / "empty.csv")


def test_an_interval_s_table_row_shows_the_point_and_its_edges() -> None:
    interval = Interval(reading="arm", measure="area", point=0.1844, low=0.0612, high=0.3331)

    assert "| arm | area | 0.184 | [0.061; 0.333] |" in interval_table([interval])
