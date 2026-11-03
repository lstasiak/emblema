"""Raising a window's readings, and reading what the encoders made of it, on known values."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from emblema.shared.kernel.tokens import Token, TokenWindow
from scripts.magnitude_response_report import (
    CONDITIONS,
    GEOMETRY,
    RESPONSES,
    SCORED,
    Responses,
    auroc,
    check_against_campaign,
    drawn_normals,
    main,
    participation_ratio,
    raised,
    readings,
    sensitivity,
    verdict,
)

# Two channels over two bins: channel 1 peaks at -4 in the second bin, channel 2 stays small, and
# a static feature that no raising touches.
WINDOW = TokenWindow.of(
    [
        Token(channel_id=9, value=5.0, time=0.0, gap=0.0, timeless=True),
        Token(channel_id=1, value=1.0, time=0.1, gap=0.0),
        Token(channel_id=1, value=0.5, time=0.6, gap=0.5),
        Token(channel_id=1, value=-4.0, time=0.8, gap=0.2),
        Token(channel_id=2, value=0.2, time=0.2, gap=0.0),
        Token(channel_id=2, value=0.3, time=0.9, gap=0.7),
    ]
)


def values_of(window: TokenWindow) -> dict[tuple[int, float], float]:
    return {
        (channel, time): value
        for channel, time, value in zip(
            window.channel_ids, window.times, window.values, strict=True
        )
    }


def test_a_cell_is_raised_alone_around_the_largest_reading() -> None:
    values = values_of(raised(WINDOW, scope="cell", factor=10.0, bins=2))

    assert values[(1, 0.6)] == 5.0
    assert values[(1, 0.8)] == -40.0
    assert values[(1, 0.1)] == 1.0
    assert values[(2, 0.9)] == 0.3
    assert values[(9, 0.0)] == 5.0


def test_a_channel_is_raised_whole_and_the_others_stay() -> None:
    values = values_of(raised(WINDOW, scope="channel", factor=3.0, bins=2))

    assert [values[(1, t)] for t in (0.1, 0.6, 0.8)] == [3.0, 1.5, -12.0]
    assert [values[(2, t)] for t in (0.2, 0.9)] == [0.2, 0.3]


def test_a_window_is_raised_reading_by_reading_but_never_its_static_features() -> None:
    values = values_of(raised(WINDOW, scope="window", factor=2.0, bins=2))

    assert values[(2, 0.2)] == pytest.approx(0.4)
    assert values[(1, 0.8)] == -8.0
    assert values[(9, 0.0)] == 5.0


def test_raising_keeps_the_window_in_canonical_order() -> None:
    # Two readings of one channel at one instant change places once one is negated by the factor.
    tied = TokenWindow.of(
        [
            Token(channel_id=1, value=-1.0, time=0.5, gap=0.0),
            Token(channel_id=1, value=3.0, time=0.5, gap=0.0),
        ]
    )

    assert raised(tied, scope="channel", factor=-1.0, bins=1).values == (-3.0, 1.0)


def test_an_unknown_scope_is_refused() -> None:
    with pytest.raises(ValueError, match="scope"):
        raised(WINDOW, scope="unit", factor=2.0, bins=2)


def test_normal_windows_are_drawn_by_the_seed_alone() -> None:
    first = drawn_normals(list(range(50)), seed=1, count=10)

    assert first == drawn_normals(list(reversed(range(50))), seed=1, count=10)
    assert first != drawn_normals(list(range(50)), seed=2, count=10)
    assert sorted(drawn_normals([3, 1], seed=1, count=10)) == [1, 3]


def test_variance_along_one_direction_spreads_over_one_and_isotropic_over_all() -> None:
    draws = np.random.default_rng(0)
    line = np.outer(draws.normal(size=500), [1.0, 2.0, 0.0])

    assert participation_ratio(line) == pytest.approx(1.0)
    assert participation_ratio(draws.normal(size=(20000, 4))) == pytest.approx(4.0, rel=0.02)


def test_the_area_counts_ties_half() -> None:
    assert auroc(np.array([2.0, 3.0]), np.array([1.0, 0.0])) == 1.0
    assert auroc(np.array([1.0]), np.array([1.0])) == 0.5
    assert auroc(np.array([0.0, 2.0]), np.array([1.0])) == 0.5


# How far each encoder's state moves at x3 and x10: three initialisations alike, and a backbone
# that moves half as far at x3 and two ninths as far at x10.
MOVES = {
    ("initialisation", 1): {3.0: 2.0, 10.0: 9.0},
    ("initialisation", 2): {3.0: 2.0, 10.0: 9.0},
    ("initialisation", 3): {3.0: 2.0, 10.0: 9.0},
    ("base1", 1): {3.0: 1.0, 10.0: 2.0},
}


def write_measured(directory: Path, *, sinking: float = 0.0) -> None:
    """Two raised windows, every distance its unraised 2 plus the state's move less ``sinking``."""
    directory.mkdir(parents=True, exist_ok=True)
    scored: list[list[object]] = [
        ["encoder", "seed", "position", "unit", "ends_at", "target", "largest", "score", "distance"]
    ]
    responses: list[list[object]] = [
        ["encoder", "seed", "position", "scope", "factor", "distance", "displacement"]
    ]
    geometry: list[list[object]] = [
        [
            "encoder",
            "seed",
            "width",
            "participation_standardised",
            "participation_raw",
            "fitting_distance",
            "state_norm",
        ]
    ]
    for (encoder, seed), moves in MOVES.items():
        for position, target, largest in ((10, 0, 2.0), (11, 0, 3.0), (12, 1, 20.0), (13, 2, 4.0)):
            score = 1.0 + target + (0.5 if encoder == "base1" else 0.0)
            scored.append([encoder, seed, position, "u", 0.0, target, largest, score, 2.0])
        geometry.append([encoder, seed, 4, 3.0, 2.0, 1.0, 5.0])
        for position in (10, 11):
            for scope in ("cell", "channel", "window"):
                for factor, move in moves.items():
                    sunk = sinking if encoder == "base1" and factor == 3.0 else 0.0
                    responses.append(
                        [encoder, seed, position, scope, factor, 2.0 + move - sunk, move]
                    )
    for name, rows in ((SCORED, scored), (RESPONSES, responses), (GEOMETRY, geometry)):
        with (directory / name).open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
    (directory / CONDITIONS).write_text(
        json.dumps({"task": "esa-mission2-events", "normal_windows": 2})
    )


def test_a_rise_is_read_against_the_other_initialisations_by_either_measure(
    tmp_path: Path,
) -> None:
    write_measured(tmp_path)
    responses = Responses.read(tmp_path)
    every = np.arange(2)

    assert responses.windows == (10, 11)
    assert responses.factors == [3.0, 10.0]
    assert responses.rise("base1", "cell", 10.0, every, "displacement") == 2.0
    # The distance rises by the move, in the unraised distance of 2.
    assert responses.rise("base1", "cell", 10.0, every, "distance") == pytest.approx(1.0)
    assert sensitivity(responses, "base1", "cell", 3.0, every, "displacement") == (0.5, True)
    assert sensitivity(responses, "base1", "cell", 10.0, every, "distance")[0] == pytest.approx(
        2.0 / 9.0
    )
    assert sensitivity(responses, "initialisation-1", "cell", 10.0, every, "displacement") == (
        1.0,
        True,
    )
    with pytest.raises(ValueError, match="measured by"):
        responses.rise("base1", "cell", 3.0, every, "area")


def test_a_backbone_falling_behind_as_the_reading_grows_is_the_size_lost(tmp_path: Path) -> None:
    write_measured(tmp_path)
    found = readings(Responses.read(tmp_path), "displacement")

    assert found[("base1", "channel")].tilt[0] == pytest.approx((2.0 / 9.0) / 0.5)
    assert found[("initialisation-1", "channel")].tilt[0] == pytest.approx(1.0)
    assert verdict(found) == "size lost on the channel scope: yes; duller at ×3: yes"


def test_a_backbone_within_the_initialisations_own_range_is_not_read_as_falling(
    tmp_path: Path,
) -> None:
    write_measured(tmp_path)
    found = readings(Responses.read(tmp_path), "displacement")
    found[("initialisation-2", "channel")] = replace(
        found[("initialisation-2", "channel")],
        tilt=(0.4, 0.3, 1.2),
        sensitivity={3.0: (0.45, 0.3, 1.1), 10.0: (0.2, 0.1, 1.0)},
    )

    assert verdict(found) == "size lost on the channel scope: no; duller at ×3: no"


def test_the_rule_is_void_when_the_draw_of_weights_alone_tilts(tmp_path: Path) -> None:
    write_measured(tmp_path)
    found = readings(Responses.read(tmp_path), "displacement")
    found[("initialisation-2", "channel")] = found[("base1", "channel")]

    assert verdict(found) == "void on the channel scope: an initialisation's own tilt falls"


def test_the_rule_is_void_when_a_rise_it_divides_is_not_above_zero(tmp_path: Path) -> None:
    # The backbone's distance at x3 sinks below the unraised one, so its tilt would be negative.
    write_measured(tmp_path, sinking=1.5)
    found = readings(Responses.read(tmp_path), "distance")

    assert not found[("base1", "channel")].positive
    assert verdict(found) == "void on the channel scope: a rise the rule divides is not above zero"


def test_the_scores_are_checked_against_the_campaign_window_by_window(tmp_path: Path) -> None:
    write_measured(tmp_path)
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["campaign", "candidate", "seed", "position", "predicted"])
        for position, score in ((10, 1.0), (11, 1.0), (12, 2.0), (13, 3.25)):
            writer.writerow(["c", "untrained_knn@reference=unit", 1, position, score])
            writer.writerow(["other", "untrained_knn", 1, position, 99.0])

    text = check_against_campaign(tmp_path, predictions, "c", "base1")

    assert "| base1 | untrained_knn | 1 | 4 of 4 | 2.50e-01 |" in text


def test_the_reading_prints_every_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_measured(tmp_path)

    main(["read", "--dir", str(tmp_path)])

    printed = capsys.readouterr().out
    for heading in ("Displacement (the declared", "Distance (desc", "size lost", "The reference"):
        assert heading in printed
    assert "| base1 | 4 | 3.0 | 2.0 | 1.000 | 2.000 | 2.00 |" in printed


def test_a_reading_without_the_rule_s_factors_or_a_second_initialisation_is_refused(
    tmp_path: Path,
) -> None:
    write_measured(tmp_path)
    responses = Responses.read(tmp_path)
    single = Responses(
        windows=responses.windows,
        distances={
            k: v
            for k, v in responses.distances.items()
            if k[0] not in ("initialisation-2", "initialisation-3")
        },
        displacements=responses.displacements,
    )
    unscaled = Responses(
        windows=responses.windows,
        distances={k: v for k, v in responses.distances.items() if k[2] != 10.0},
        displacements=responses.displacements,
    )

    with pytest.raises(ValueError, match="at least two"):
        readings(single, "displacement")
    with pytest.raises(ValueError, match="×10"):
        readings(unscaled, "displacement")
