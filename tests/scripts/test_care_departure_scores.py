"""The channel-local departures over an excerpt's hours: what is scored and from what."""

import numpy as np
import pytest

from scripts.care_departure_scores import (
    AVAILABILITY,
    FAMILY,
    FIRST_DAYS,
    TEMPERATURES,
    arrays_of,
    departure_scores,
    excerpt_scores,
    first_days,
    names_of_family,
)
from scripts.care_excerpts import Excerpt, Sensors
from tests.scripts.test_care_excerpts import excerpt

HOURS, KNOWN = 400, 300


# Enough for the readings' departures over 72 hours, which cost most, to be read twice.
FEW, FEW_KNOWN = 110, 80


def readings(seed: int = 1, hours: int = HOURS) -> np.ndarray:
    """Hours of six readings a channel, each channel a level with its own noise."""
    draws = np.random.default_rng(seed)
    levels = np.array([20.0, 500.0, 0.3])
    return levels[None, :, None] * (1.0 + 0.01 * draws.standard_normal((hours, 3, 6)))


def test_the_family_is_the_twelve_detectors_of_the_telemetrys_verdict() -> None:
    scores = departure_scores(readings(hours=FEW), FEW_KNOWN)

    assert sorted(scores) == sorted(names_of_family())
    assert len(scores) == 12
    assert "max-window-6h" in scores
    assert "mean-bins-72h" in scores


def test_a_step_in_one_channel_departs_and_the_hours_before_it_do_not() -> None:
    stepped = readings()
    stepped[350:, 0] *= 1.2

    scores = departure_scores(stepped, KNOWN)["max-window-24h"]

    assert scores[350] > 10 * np.nanmedian(scores[:350])
    plain = departure_scores(readings(), KNOWN)["max-window-24h"]
    assert np.array_equal(scores[:350], plain[:350], equal_nan=True)


def test_a_channels_unit_does_not_move_another_channels_score() -> None:
    plain = readings(hours=FEW)
    scaled = plain.copy()
    scaled[:, 1] /= 100.0
    rescaled = departure_scores(scaled, FEW_KNOWN)

    for arm, scores in departure_scores(plain, FEW_KNOWN).items():
        assert rescaled[arm] == pytest.approx(scores, nan_ok=True), arm


def test_nothing_after_the_known_hours_enters_the_scale_or_the_floor() -> None:
    plain = readings(hours=FEW)
    later = plain.copy()
    later[FEW_KNOWN:] *= 3.0
    moved = departure_scores(later, FEW_KNOWN)

    for arm, scores in departure_scores(plain, FEW_KNOWN).items():
        assert np.array_equal(moved[arm][:FEW_KNOWN], scores[:FEW_KNOWN], equal_nan=True), arm


def test_an_hour_that_is_not_whole_is_not_scored_and_starts_no_departure() -> None:
    plain = readings()
    holed = plain.copy()
    holed[200] = np.nan

    scores = departure_scores(holed, KNOWN)["max-window-6h"]

    assert np.isnan(scores[200])
    # The hours after the hole read a history one hour shorter and nothing else.
    assert np.nanmax(scores[201:210]) < 2 * np.nanmax(
        departure_scores(plain, KNOWN)["max-window-6h"]
    )


def test_an_hour_is_scored_once_half_its_history_holds_a_reading() -> None:
    stopped = readings()
    stopped[100:200] = np.nan

    for span in (6, 24, 72):
        scores = departure_scores(stopped, KNOWN)[f"max-window-{span}h"]
        needed = -(-(span - 1) // 2)

        assert np.isnan(scores[:needed]).all()
        assert not np.isnan(scores[needed])
        assert np.isnan(scores[200 : 200 + needed]).all()
        assert not np.isnan(scores[200 + needed])


def test_the_mean_over_channels_is_moved_less_by_one_channel_than_the_largest() -> None:
    stepped = readings()
    stepped[350, 0] *= 1.2
    scores = departure_scores(stepped, KNOWN)

    assert scores["mean-window-24h"][350] < scores["max-window-24h"][350]


def twelve_days(seed: int = 4) -> tuple[np.ndarray, Sensors]:
    """Twelve days of three channels, the last two the scored window."""
    values = 50.0 + np.random.default_rng(seed).standard_normal((6 * 24 * 12, 3))
    columns = tuple(f"sensor_{column}_avg" for column in range(3))
    return values, Sensors(channels=columns, temperatures=columns[:2], conditions=columns[2:])


def read(values: np.ndarray, status: np.ndarray | None = None, farm: str = "B") -> Excerpt:
    rows = len(values)
    return excerpt(
        values,
        fitted_rows=rows - 6 * 48,
        event_rows=(rows - 6 * 40, rows - 1),
        status=status,
        farm=farm,
    )


def test_an_excerpt_is_scored_by_every_family_beside_the_answer_with_no_model() -> None:
    values, sensors = twelve_days()
    status = np.zeros(len(values), dtype=np.int64)
    status[600:603] = 4

    hours, scores = excerpt_scores(read(values, status), sensors)

    families = {name.split("/")[0] for name in scores}
    assert families == {
        AVAILABILITY,
        FAMILY,
        TEMPERATURES,
        *(first_days(days) for days in FIRST_DAYS),
    }
    assert all(len(series) == hours.count for series in scores.values())
    # The verdict's family reads both departures; the families beside it the summary's alone.
    assert sum(name.startswith(f"{FAMILY}/") for name in scores) == 12
    assert sum(name.startswith(f"{TEMPERATURES}/") for name in scores) == 6
    assert scores[AVAILABILITY][100] == 0.5
    assert np.isnan(scores[f"{FAMILY}/max-window-6h"][100])
    assert not np.isnan(scores[AVAILABILITY]).any()


def test_the_first_farms_stopped_hours_are_scored_and_still_counted_as_stops() -> None:
    values, sensors = twelve_days()
    status = np.zeros(len(values), dtype=np.int64)
    stopped = slice(len(values) - 6 * 30, len(values) - 6 * 20)
    status[stopped] = 4
    hour = (len(values) - 6 * 25) // 6

    _, first = excerpt_scores(read(values, status, farm="A"), sensors)
    _, other = excerpt_scores(read(values, status, farm="B"), sensors)

    arm = f"{FAMILY}/max-window-6h"
    assert not np.isnan(first[arm][hour])
    assert np.isnan(other[arm][hour])
    assert first[AVAILABILITY][hour] == other[AVAILABILITY][hour] == 1.0


def test_an_excerpts_file_holds_the_rows_read_not_their_status() -> None:
    values, sensors = twelve_days()
    status = np.zeros(len(values), dtype=np.int64)
    status[len(values) - 6 * 30 : len(values) - 6 * 20] = 4
    first = read(values, status, farm="A")
    hours, scores = excerpt_scores(first, sensors)

    arrays = arrays_of(first, hours, scores)

    assert np.array_equal(arrays["row_normal"], first.read)
    assert not np.array_equal(arrays["row_normal"], first.normal)
    assert arrays["scores"].shape == (len(scores), hours.count)
    assert arrays["arms"].tolist() == sorted(scores)
    assert (int(arrays["fitted"]), int(arrays["first"]), int(arrays["last"])) == (
        hours.fitted,
        hours.first,
        hours.last,
    )


def test_no_family_takes_its_scale_or_floor_from_the_scored_window() -> None:
    values, sensors = twelve_days()
    later = values.copy()
    later[len(values) - 6 * 48 :] *= 3.0

    hours, plain = excerpt_scores(read(values), sensors)
    _, moved = excerpt_scores(read(later), sensors)

    for name, series in plain.items():
        assert np.array_equal(
            moved[name][: hours.fitted], series[: hours.fitted], equal_nan=True
        ), name


def test_a_turbines_first_days_alone_set_the_scale_of_the_families_named_by_them() -> None:
    values, sensors = twelve_days()
    week = 7 * 24
    later = values.copy()
    later[6 * week : len(values) - 6 * 48] *= 3.0

    _, plain = excerpt_scores(read(values), sensors)
    _, moved = excerpt_scores(read(later), sensors)

    arm = "max-window-6h"
    seven, whole = f"{first_days(7)}/{arm}", f"{FAMILY}/{arm}"
    assert np.array_equal(moved[seven][:week], plain[seven][:week], equal_nan=True)
    # The whole year's scale reads the days after the first seven, and so does thirty days'
    # of a year ten days long.
    assert not np.array_equal(moved[whole][:week], plain[whole][:week], equal_nan=True)
    thirty = f"{first_days(30)}/{arm}"
    assert not np.array_equal(moved[thirty][:week], plain[thirty][:week], equal_nan=True)
