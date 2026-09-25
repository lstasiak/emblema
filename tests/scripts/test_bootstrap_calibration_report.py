"""The calibration read off a few small datasets, stored and rendered; nothing is trained."""

from pathlib import Path

import pytest

from scripts.bootstrap_calibration_report import (
    CALIBRATION,
    Calibration,
    calibrate,
    main,
    read,
    render,
    write,
)


def test_the_rates_are_shares_of_the_datasets_drawn() -> None:
    found = calibrate(18, datasets=20, resamples=100)

    assert (found.units, found.datasets, found.resamples) == (18, 20, 100)
    assert 0.0 <= found.false_positive <= 1.0
    assert 0.0 <= found.one_sided_false_positive <= found.false_positive
    assert found.coverage > 0.5


def test_the_calibration_is_written_and_read_back_as_it_was(tmp_path: Path) -> None:
    rows = [calibrate(units, datasets=5, resamples=50) for units in (6, 12)]

    written = write(rows, tmp_path / "calibration")

    assert written.name == CALIBRATION
    assert read(tmp_path / "calibration") == tuple(rows)


def test_the_table_has_a_row_per_unit_count_with_the_rates_as_percentages() -> None:
    rows = [
        Calibration(
            units=21,
            datasets=400,
            resamples=2000,
            coverage=0.917,
            false_positive=0.095,
            one_sided_false_positive=0.05,
        )
    ]

    rendered = render(rows)

    assert "| 21 | 91.7% | 9.5% | 5.0% |" in rendered
    assert "2,000 resamples; 400 datasets" in rendered


def test_the_report_runs_end_to_end_and_renders_again_from_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--out", str(tmp_path / "out"), "--datasets", "3", "--resamples", "20"])
    first = capsys.readouterr().out
    main(["--report-only", str(tmp_path / "out")])
    again = capsys.readouterr().out

    assert first == again
    assert "| 80 |" in first


def test_a_run_without_a_direction_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--out DIR or --report-only DIR"):
        main([])
    with pytest.raises(SystemExit, match="nothing to render"):
        main(["--report-only", str(tmp_path)])
