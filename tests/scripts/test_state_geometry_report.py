"""The scorers over an encoder's states, checked on states whose geometry is known."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from scripts.state_geometry_report import (
    CONDITIONS,
    DISTANCE,
    FITTING_UNITS,
    MAHALANOBIS,
    MONTH,
    PER_UNIT,
    RESIDUAL,
    TOP,
    WHITENED,
    WINDOWS,
    Measured,
    RankedScores,
    Reading,
    auroc,
    check_against_campaign,
    enters,
    main,
    scorers_of,
)

Floats = NDArray[np.float64]

# Four directions whose normal variation shrinks from the first to the last, turned by a fixed
# rotation so the differences live in the correlations between features, as they do in an
# encoder's states, and survive the detector's standardisation of each feature.
SPREAD = np.array([10.0, 3.0, 1.0, 0.3])
ROTATION, _ = np.linalg.qr(np.random.default_rng(42).normal(size=(4, 4)))
LOUD = ROTATION[:, 0]
QUIET = ROTATION[:, 3]


def normal_states(draws: np.random.Generator, count: int) -> Floats:
    states: Floats = (draws.normal(size=(count, 4)) * SPREAD) @ ROTATION.T
    return states


def test_the_area_counts_ties_half_and_weights_as_repeats() -> None:
    scores = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    positive = np.array([False, False, True, True, False])
    ranked = RankedScores.of(scores, positive)

    # Pairs: the event at 1 beats the normal at 0, ties the one at 1, loses to 3; the event at 2
    # beats two and loses to one.
    assert ranked.auroc() == pytest.approx((1 + 0.5 + 0 + 1 + 1 + 0) / (2 * 3))
    # Weighting a window twice reads as listing it twice.
    weights = np.array([1.0, 2.0, 1.0, 1.0, 1.0])
    repeated = auroc(np.r_[scores, 1.0], np.r_[positive, False])
    assert ranked.auroc(weights) == pytest.approx(repeated)
    assert np.isnan(ranked.auroc(np.array([1.0, 1.0, 0.0, 0.0, 1.0])))


def sides(
    *, shift: Floats, drifted_month: Floats | None = None, events: int = 40, months: int = 4
) -> tuple[Floats, Floats, list[str], list[str], NDArray[np.bool_]]:
    """Normal states, and a scored side of ``months`` with ``events`` windows moved by ``shift``.

    ``drifted_month`` moves every window of the last month, event or not.
    """
    draws = np.random.default_rng(7)
    fitting = normal_states(draws, 2000)
    per_month = 60
    scored = normal_states(draws, months * per_month)
    units = [f"m{month}" for month in range(months) for _ in range(per_month)]
    positive = np.zeros(len(scored), dtype=bool)
    positive[draws.choice(len(scored), events, replace=False)] = True
    scored[positive] += shift
    if drifted_month is not None:
        scored[np.asarray(units) == f"m{months - 1}"] += drifted_month
    return fitting, scored, ["f"] * len(fitting), units, positive


def test_events_off_a_quiet_direction_are_read_by_the_residual_and_by_whitening() -> None:
    fitting, scored, fitting_units, units, positive = sides(shift=2.0 * QUIET)

    scores, profile = scorers_of(
        fitting, scored, fitting_units=fitting_units, scored_units=units, positive=positive
    )

    assert profile.top <= 2
    assert profile.separation[-1] > 0.9 > profile.separation[0]
    assert auroc(scores[RESIDUAL], positive) > 0.9 > auroc(scores[TOP], positive)
    assert auroc(scores[WHITENED], positive) > auroc(scores[DISTANCE], positive) + 0.02
    assert auroc(scores[WHITENED], positive) > 0.99
    assert auroc(scores[MAHALANOBIS], positive) > 0.99


def test_events_along_the_loudest_direction_are_read_by_the_top_and_not_helped_by_whitening() -> (
    None
):
    fitting, scored, fitting_units, units, positive = sides(shift=25.0 * LOUD)

    scores, profile = scorers_of(
        fitting, scored, fitting_units=fitting_units, scored_units=units, positive=positive
    )

    # The detector's standardisation reorders the components, so the loud direction is read in
    # the top block rather than at any one index.
    assert profile.separation[: profile.top].max() > 0.9 > profile.separation[profile.top :].max()
    assert auroc(scores[TOP], positive) > 0.9 > auroc(scores[RESIDUAL], positive)
    assert auroc(scores[WHITENED], positive) <= auroc(scores[DISTANCE], positive) + 0.02


def test_a_drifted_month_is_read_by_the_relative_distance_and_the_own_unit_reference() -> None:
    fitting, scored, fitting_units, units, positive = sides(
        shift=2.0 * QUIET, drifted_month=60.0 * LOUD
    )

    scores, _ = scorers_of(
        fitting, scored, fitting_units=fitting_units, scored_units=units, positive=positive
    )

    # Under the pooled Gaussian the drifted month's normal windows outscore the other months'
    # events; against its own month's centre they do not.
    assert auroc(scores[MONTH], positive) > auroc(scores[MAHALANOBIS], positive) + 0.04
    assert auroc(scores[MONTH], positive) > auroc(scores[DISTANCE], positive) + 0.1
    # The campaign's own-unit reference rescales one distance a month; it cannot undo a drift
    # that dominates every distance of the month, so it gains less.
    assert auroc(scores[PER_UNIT], positive) > auroc(scores[DISTANCE], positive)


def test_a_month_too_thin_for_a_centre_of_its_own_is_read_against_the_fitting_side() -> None:
    fitting, scored, fitting_units, units, positive = sides(shift=np.zeros(4))
    units[0] = "lonely"

    scores, _ = scorers_of(
        fitting, scored, fitting_units=fitting_units, scored_units=units, positive=positive
    )

    assert scores[MONTH][0] == pytest.approx(scores[MAHALANOBIS][0])
    assert scores[MONTH][1] != pytest.approx(scores[MAHALANOBIS][1])


def write_measured(directory: Path, *, pretrained: int = 2) -> None:
    """Two initialisations whose states hold no signal, and backbones whose quiet direction does."""
    directory.mkdir(parents=True, exist_ok=True)
    fitting, scored, fitting_units, units, positive = sides(shift=1.0 * QUIET, months=8)
    for seed in (1, 2):
        plain = np.random.default_rng(seed)
        np.savez(
            directory / f"states-initialisation-{seed}.npz",
            fitting=normal_states(plain, len(fitting)),
            scored=normal_states(plain, len(scored)),
        )
    for index in range(pretrained):
        np.savez(directory / f"states-base{index + 1}.npz", fitting=fitting, scored=scored + index)
    with (directory / WINDOWS).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["position", "unit", "ends_at", "target", "largest"])
        event = 0
        for position, (unit, is_event) in enumerate(zip(units, positive, strict=True)):
            event += int(is_event)
            writer.writerow([position, unit, float(position), event if is_event else 0, 2.0])
    with (directory / FITTING_UNITS).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["unit"])
        writer.writerows([[unit] for unit in fitting_units])
    (directory / CONDITIONS).write_text(json.dumps({"task": "esa-mission2-events"}))


def test_the_whitened_distance_enters_when_both_backbones_gain_and_no_initialisation_does(
    tmp_path: Path,
) -> None:
    write_measured(tmp_path)
    reading = Reading.of(Measured.read(tmp_path), control=DISTANCE, resamples=50, seed=1)

    assert reading.measured.pretrained == ["base1", "base2"]
    assert reading.ruled == (WHITENED, MONTH, PER_UNIT)
    assert enters(reading, WHITENED) == "yes"
    # The own-unit reference reads the same distance a month at a time: no whitening, no gain.
    assert enters(reading, PER_UNIT) == "no"
    with pytest.raises(ValueError, match="control"):
        enters(reading, DISTANCE)


def test_the_control_is_the_mission_s_own_campaign_detector(tmp_path: Path) -> None:
    write_measured(tmp_path)

    reading = Reading.of(Measured.read(tmp_path), control=PER_UNIT, resamples=20, seed=1)

    assert reading.ruled == (WHITENED, MONTH, DISTANCE)
    assert all(scorer != PER_UNIT for _, scorer in reading.gains)
    with pytest.raises(ValueError, match="control is one of"):
        Reading.of(Measured.read(tmp_path), control=WHITENED, resamples=20, seed=1)


def test_the_rule_is_void_with_one_backbone(tmp_path: Path) -> None:
    write_measured(tmp_path, pretrained=1)
    reading = Reading.of(Measured.read(tmp_path), control=DISTANCE, resamples=20, seed=1)

    assert enters(reading, WHITENED).startswith("void")


def test_the_reading_prints_every_table_and_writes_the_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_measured(tmp_path)

    main(["read", "--dir", str(tmp_path), "--resamples", "20"])

    printed = capsys.readouterr().out
    for heading in ("Where the events' signal sits", "Areas", "Gains over", "enters the screen"):
        assert heading in printed
    with (tmp_path / "profile.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 * 4
    assert (tmp_path / "scores.csv").exists()


def test_the_scores_are_checked_against_the_campaign_window_by_window(tmp_path: Path) -> None:
    write_measured(tmp_path)
    reading = Reading.of(Measured.read(tmp_path), control=DISTANCE, resamples=10, seed=1)
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["campaign", "candidate", "seed", "position", "predicted"])
        for position, score in zip(
            reading.measured.positions, reading.scores["base1"][DISTANCE], strict=True
        ):
            writer.writerow(["c", "frozen_knn", 1, int(position), score + 0.5])
            writer.writerow(["other", "frozen_knn", 1, int(position), 99.0])

    text = check_against_campaign(reading, predictions, "c", "base1", DISTANCE)

    assert "| base1 | frozen_knn | 1 | 480 of 480 | 5.00e-01 | 1.000000 |" in text
