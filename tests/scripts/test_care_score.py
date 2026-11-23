"""The fault benchmark's score: its counter, its four parts and the rules that override them."""

from fractions import Fraction

import numpy as np
import pytest

from scripts.care_score import CareScore, ScoredExcerpt, criticality, f_score

ROWS = 400
FIRST, LAST = 100, 299


def flags(*runs: tuple[int, int]) -> np.ndarray:
    flagged = np.zeros(ROWS, dtype=np.bool_)
    for start, stop in runs:
        flagged[start:stop] = True
    return flagged


def scored(
    flagged: np.ndarray, *, anomaly: bool, normal: np.ndarray | None = None
) -> ScoredExcerpt:
    return CareScore().excerpt(
        event=1,
        anomaly=anomaly,
        flagged=flagged,
        normal=np.ones(ROWS, dtype=np.bool_) if normal is None else normal,
        first=FIRST,
        last=LAST,
    )


def test_the_counter_rises_with_a_flag_eases_without_and_waits_out_another_status() -> None:
    flagged = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0], dtype=np.bool_)
    normal = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=np.bool_)

    assert criticality(flagged, normal).tolist() == [1, 2, 1, 0, 0, 1, 1, 1, 1]


def test_a_detector_that_flags_nothing_scores_nought() -> None:
    silent = flags()

    reading = CareScore().of([scored(silent, anomaly=True), scored(silent, anomaly=False)])

    assert reading.accuracy == 1.0
    assert reading.found == 0
    assert reading.score == 0.0


def test_a_detector_that_flags_everything_scores_its_accuracy_of_nought() -> None:
    loud = flags((0, ROWS))

    reading = CareScore().of([scored(loud, anomaly=True), scored(loud, anomaly=False)])

    assert (reading.found, reading.falsely_found) == (1, 1)
    assert reading.earliness == 1.0
    assert reading.score == 0.0


def test_flags_drawn_by_a_coin_score_about_a_half() -> None:
    # Windows as long as the benchmark's: over a few hundred rows a coin never holds the counter
    # up for 72, finds nothing and scores nought.
    rows, first, last = 3000, 300, 2699
    draws = np.random.default_rng(3)
    excerpts = [
        CareScore().excerpt(
            event=number,
            anomaly=bool(number % 2),
            flagged=draws.random(rows) < 0.5,
            normal=np.ones(rows, dtype=np.bool_),
            first=first,
            last=last,
        )
        for number in range(100)
    ]

    assert CareScore().of(excerpts).score == pytest.approx(0.5, abs=0.05)


def test_the_score_is_the_mean_of_its_parts_with_the_accuracy_counted_twice() -> None:
    fault = scored(flags((100, 300)), anomaly=True)
    quiet = scored(flags((0, 40)), anomaly=False)

    reading = CareScore().of([fault, quiet])

    assert (reading.coverage, reading.earliness, reading.reliability) == (1.0, 1.0, 1.0)
    assert reading.accuracy == pytest.approx(0.9)
    assert reading.score == pytest.approx((1.0 + 1.0 + 1.0 + 2 * 0.9) / 5)


def test_an_accuracy_of_a_half_or_less_is_the_whole_score() -> None:
    fault = scored(flags((100, 300)), anomaly=True)
    quiet = scored(flags((0, 200)), anomaly=False)

    assert CareScore().of([fault, quiet]).score == 0.5


def test_a_fault_is_found_once_the_counter_reaches_its_level_and_not_one_row_sooner() -> None:
    short = scored(flags((100, 171)), anomaly=True)
    enough = scored(flags((100, 172)), anomaly=True)
    quiet = scored(flags(), anomaly=False)

    assert CareScore().of([short, quiet]).found == 0
    assert CareScore().of([enough, quiet]).found == 1


def test_flags_between_the_windows_start_and_the_event_count_towards_finding_it() -> None:
    before = scored(flags((0, 80)), anomaly=True)

    assert before.most_critical == 80
    assert before.coverage == 0.0


def test_flags_after_the_event_do_not_find_it() -> None:
    assert scored(flags((300, 400)), anomaly=True).most_critical == 0


def test_rows_of_another_status_are_neither_right_nor_wrong() -> None:
    normal = np.ones(ROWS, dtype=np.bool_)
    normal[150:250] = False
    flagged = flags((100, 150))

    fault = scored(flagged, anomaly=True, normal=normal)

    # Fifty flagged of the hundred normal rows of the event, none flagged outside it.
    assert fault.coverage == pytest.approx(f_score(found=50, missed=50, false=0, beta=0.5))
    assert fault.accuracy == pytest.approx(250 / 300)


def test_the_earliness_reads_every_row_of_the_event_whatever_its_status() -> None:
    normal = np.zeros(ROWS, dtype=np.bool_)

    assert scored(flags((100, 300)), anomaly=True, normal=normal).earliness == 1.0


def test_a_flag_weighs_one_through_the_first_half_and_less_towards_the_fault() -> None:
    weights = CareScore().weights(4)

    assert weights.tolist() == pytest.approx([1.0, 1.0, 6 / 7, 2 / 7])


def test_the_descent_can_start_at_a_quarter_as_the_publishers_default_does() -> None:
    weights = CareScore(descends_from=Fraction(1, 4)).weights(8)

    assert weights[:2].tolist() == [1.0, 1.0]
    assert np.all(np.diff(weights[2:]) < 0.0)


def test_early_flags_score_more_than_late_ones() -> None:
    early = scored(flags((100, 150)), anomaly=True).earliness
    late = scored(flags((250, 300)), anomaly=True).earliness

    assert early > 2 * late


def test_an_excerpt_without_a_fault_has_no_coverage_and_no_earliness() -> None:
    quiet = scored(flags((0, 10)), anomaly=False)

    assert np.isnan(quiet.coverage)
    assert np.isnan(quiet.earliness)


def test_excerpts_of_one_label_are_refused() -> None:
    with pytest.raises(ValueError, match="with a fault and one without"):
        CareScore().of([scored(flags(), anomaly=True)])


def test_an_event_outside_the_scored_window_is_refused() -> None:
    with pytest.raises(ValueError, match="lie outside"):
        CareScore().excerpt(
            event=1,
            anomaly=True,
            flagged=flags(),
            normal=np.ones(ROWS, dtype=np.bool_),
            first=350,
            last=400,
        )


def test_flags_and_statuses_of_different_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="flags were given"):
        CareScore().excerpt(
            event=1,
            anomaly=True,
            flagged=flags(),
            normal=np.ones(ROWS - 1, dtype=np.bool_),
            first=FIRST,
            last=LAST,
        )
