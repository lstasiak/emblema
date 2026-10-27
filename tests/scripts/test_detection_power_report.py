"""The detection rule's power is simulated with the harness's own sweep, on a known structure."""

from pathlib import Path

import numpy as np
import pytest

from scripts.detection_power_report import (
    POWER,
    STRUCTURE,
    Structure,
    average_precision,
    check_against_the_harness,
    confirms,
    main,
)


def write_structure(path: Path, units: int, windows: int, events: int) -> Path:
    """``units`` units of ``windows`` windows, the first ``events`` units holding one event each."""
    lines = ["unit,position,event"]
    position = 0
    for unit in range(units):
        for _ in range(windows):
            lines.append(f"u{unit:02d},{position},{unit + 1 if unit < events else 0}")
            position += 1
    path.write_text("\n".join(lines) + "\n")
    return path


def test_the_sweep_agrees_with_the_harness() -> None:
    check_against_the_harness()


def test_an_event_weighs_one_however_many_windows_it_spans() -> None:
    structure = Structure(
        units=np.array([0, 0, 0, 1, 1]), events=np.array([1, 1, 1, 0, 2]), unit_count=2
    )

    assert structure.weights.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 1.0, 1.0])
    assert [stratum.tolist() for stratum in structure.strata] == [[0, 1], []]
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    # The three-window event is found whole before the normal window, the one-window event
    # after it: the mean of a precision of one and of two thirds, each event weighing one.
    area = average_precision(scores, structure.events, structure.weights, np.ones(5))
    assert area == pytest.approx((1.0 + 2 / 3) / 2)


def test_a_candidate_no_better_than_the_control_is_not_confirmed(tmp_path: Path) -> None:
    structure = Structure.read(write_structure(tmp_path / "s.csv", units=12, windows=4, events=4))
    draws = np.random.default_rng(1)
    control = draws.standard_normal(48)

    assert not confirms(structure, control, control, minimum=0.02, resamples=50, draws=draws)


def test_the_report_writes_the_power_per_level_and_gain(tmp_path: Path) -> None:
    structure = write_structure(tmp_path / "s.csv", units=12, windows=4, events=4)

    main(
        [
            "simulate",
            "--structure",
            str(structure),
            "--out",
            str(tmp_path),
            "--level",
            "0.3",
            "--gains",
            "0.1",
            "--worlds",
            "4",
            "--resamples",
            "20",
        ]
    )

    written = (tmp_path / POWER).read_text().splitlines()
    assert written[0] == "level,gain,minimum,worlds,power"
    assert written[1].startswith("0.3,0.1,0.02,4,")
    assert STRUCTURE == "structure.csv"
