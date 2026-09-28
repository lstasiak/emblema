"""The calibration over rankings read off a few small datasets, stored and rendered."""

from pathlib import Path

import pytest

from scripts.ranking_calibration_report import (
    CALIBRATION,
    RankingCalibration,
    Setting,
    calibrate,
    main,
    read,
    render,
    write,
)

SMALL = {"units": 400, "positives": 56, "datasets": 12, "resamples": 60}


def test_the_rates_are_shares_of_the_datasets_drawn() -> None:
    (found,) = calibrate([Setting(level=0.80, repeats=2, spread=0.0)], **SMALL)

    assert (found.units, found.positives, found.datasets, found.resamples) == (400, 56, 12, 60)
    assert (found.level, found.gain, found.repeats, found.spread) == (0.80, 0.02, 2, 0.0)
    assert 0.0 <= found.one_sided_false_positive <= found.false_positive <= 1.0
    assert found.coverage > 0.5
    assert found.seconds > 0.0


def test_splitting_the_datasets_over_processes_finds_the_same_rates() -> None:
    settings = [Setting(level=0.75, repeats=1, spread=0.01)]

    alone = calibrate(settings, **SMALL)
    shared = calibrate(settings, **SMALL, workers=2)

    assert [(r.coverage, r.false_positive, r.one_sided_false_positive) for r in alone] == [
        (r.coverage, r.false_positive, r.one_sided_false_positive) for r in shared
    ]


def test_a_spread_of_repeats_the_units_cannot_see_makes_zero_look_excluded_more_often() -> None:
    steady, moving = calibrate(
        [Setting(level=0.80, repeats=1, spread=0.0), Setting(level=0.80, repeats=1, spread=0.1)],
        units=400,
        positives=56,
        datasets=40,
        resamples=100,
    )

    assert moving.false_positive > steady.false_positive


def test_the_calibration_is_written_and_read_back_as_it_was(tmp_path: Path) -> None:
    rows = calibrate([Setting(level=0.70, repeats=1, spread=0.0)], **SMALL)

    written = write(rows, tmp_path / "calibration")

    assert written.name == CALIBRATION
    assert read(tmp_path / "calibration") == rows


def test_the_table_has_a_row_per_setting_with_the_rates_as_percentages() -> None:
    rows = [
        RankingCalibration(
            level=0.85,
            gain=0.02,
            repeats=5,
            spread=0.03,
            units=3994,
            positives=568,
            datasets=400,
            resamples=1000,
            coverage=0.712,
            false_positive=0.31,
            one_sided_false_positive=0.155,
            seconds=5.1,
        )
    ]

    rendered = render(rows)

    assert "| 0.85 | 5 | 0.03 | 71.2% | 31.0% | 15.5% | 5.10 |" in rendered
    assert "3,994 stays, 568 positive" in rendered
    assert "1,000 resamples in two strata; 400 datasets" in rendered


def test_the_report_runs_end_to_end_and_renders_again_from_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scripts.ranking_calibration_report.UNITS", 200)
    monkeypatch.setattr("scripts.ranking_calibration_report.POSITIVES", 28)
    main(["--out", str(tmp_path / "out"), "--datasets", "2", "--resamples", "10"])
    first = capsys.readouterr().out
    main(["--report-only", str(tmp_path / "out")])
    again = capsys.readouterr().out

    assert first == again
    assert "| 0.70 | 1 | 0 |" in first
    assert "| 0.85 | 5 | 0.03 |" in first


def test_a_run_without_a_direction_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--out DIR or --report-only DIR"):
        main([])
    with pytest.raises(SystemExit, match="nothing to render"):
        main(["--report-only", str(tmp_path)])


def test_a_run_over_no_process_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--workers must be at least one, got 0"):
        main(["--out", str(tmp_path), "--workers", "0"])
