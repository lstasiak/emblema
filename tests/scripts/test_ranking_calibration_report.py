"""The calibration over rankings read off a few small datasets, stored and rendered."""

import csv
from pathlib import Path

import pytest

from scripts.ranking_calibration_report import (
    CALIBRATION,
    DECOMPOSITION,
    Decomposition,
    RankingCalibration,
    Setting,
    calibrate,
    decompose,
    main,
    read,
    read_decomposition,
    render,
    render_decomposition,
    suite,
    write,
    write_decomposition,
)

SMALL = {"units": 400, "positives": 56, "datasets": 12, "resamples": 60}


def test_the_rates_are_shares_of_the_datasets_drawn() -> None:
    (found,) = calibrate([Setting(level=0.80, repeats=2, spread=0.0)], **SMALL)

    assert (found.units, found.positives, found.datasets, found.resamples) == (400, 56, 12, 60)
    assert (found.level, found.repeats, found.spread) == (0.80, 2, 0.0)
    assert (found.shared, found.levels) == (0.5, 0)
    assert found.gain == pytest.approx(0.02)
    assert 0.0 <= found.one_sided_false_positive <= found.false_positive <= 1.0
    assert found.coverage > 0.5
    assert found.seconds > 0.0


def test_answers_in_few_levels_are_read_against_the_gain_those_levels_can_reach() -> None:
    (found,) = calibrate([Setting(level=0.80, repeats=1, spread=0.0, levels=4)], **SMALL)

    assert found.levels == 4
    assert found.gain < 0.02


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


def test_the_registered_suite_crosses_both_levels_with_every_regime() -> None:
    settings = suite("registered")

    assert len(settings) == 8
    assert {(s.level, s.repeats, s.spread) for s in settings} >= {(0.70, 1, 0.0), (0.85, 5, 0.03)}
    assert all((s.shared, s.levels) == (0.5, 0) for s in settings)


def test_the_robustness_suite_moves_one_assumption_at_a_time_without_a_spread() -> None:
    settings = suite("robustness")

    assert all((s.repeats, s.spread) == (5, 0.0) for s in settings)
    assert {(s.level, s.shared, s.levels) for s in settings} == {
        (0.60, 0.5, 0),
        (0.80, 0.2, 0),
        (0.80, 0.9, 0),
        (0.80, 0.5, 20),
        (0.80, 0.5, 5),
    }


def test_a_suite_of_no_known_name_is_refused() -> None:
    with pytest.raises(SystemExit, match="no suite named 'other'"):
        suite("other")


def test_the_decomposition_splits_the_spread_off_the_variance_over_stays() -> None:
    (found,) = decompose(
        [Setting(level=0.80, repeats=5, spread=0.0), Setting(level=0.80, repeats=5, spread=0.05)],
        units=400,
        positives=56,
        datasets=60,
    )

    assert (found.level, found.repeats, found.spread, found.datasets) == (0.80, 5, 0.05, 60)
    assert found.over_both > found.over_stays
    assert found.over_seeds == pytest.approx(found.expected_over_seeds, rel=0.35)
    assert 0.0 < found.predicted_coverage < 0.95


def test_the_decomposition_over_processes_finds_the_same_spreads() -> None:
    settings = [Setting(level=0.80, repeats=2, spread=0.03)]

    alone = decompose(settings, units=200, positives=28, datasets=10)
    shared = decompose(settings, units=200, positives=28, datasets=10, workers=2)

    assert alone == shared


def test_a_decomposition_with_no_spread_to_split_has_no_rows() -> None:
    assert decompose([Setting(level=0.80, repeats=5, spread=0.0)], units=200, positives=28) == ()


def test_the_prediction_of_coverage_is_the_normal_interval_of_the_stays_against_both() -> None:
    same = Decomposition(
        level=0.8, repeats=5, spread=0.0, datasets=1, over_stays=0.01, over_both=0.01
    )
    doubled = Decomposition(
        level=0.8, repeats=5, spread=0.01, datasets=1, over_stays=0.01, over_both=0.02
    )

    assert same.predicted_coverage == pytest.approx(0.95)
    assert doubled.predicted_coverage == pytest.approx(0.6729, abs=1e-4)
    assert doubled.over_seeds == pytest.approx(0.01 * 3**0.5)
    assert doubled.expected_over_seeds == pytest.approx(0.01 * (2 / 5) ** 0.5)


def test_the_tables_are_written_and_read_back_as_they_were(tmp_path: Path) -> None:
    rows = calibrate([Setting(level=0.70, repeats=1, spread=0.0, shared=0.9)], **SMALL)
    parts = decompose([Setting(level=0.70, repeats=2, spread=0.02)], units=200, positives=28)

    assert write(rows, tmp_path).name == CALIBRATION
    assert write_decomposition(parts, tmp_path).name == DECOMPOSITION
    assert read(tmp_path) == rows
    assert read_decomposition(tmp_path) == parts


def test_a_calibration_written_before_the_share_and_levels_columns_reads_as_the_defaults(
    tmp_path: Path,
) -> None:
    columns = [c for c in RankingCalibration.COLUMNS if c not in ("shared", "levels")]
    with (tmp_path / CALIBRATION).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(dict.fromkeys(columns, "1") | {"level": "0.7", "gain": "0.02"})

    (found,) = read(tmp_path)

    assert (found.shared, found.levels) == (0.5, 0)


def test_the_calibration_table_has_a_row_per_setting_with_the_rates_as_percentages() -> None:
    rows = [
        RankingCalibration(
            level=0.85,
            gain=0.0183,
            repeats=5,
            spread=0.0,
            shared=0.5,
            levels=5,
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

    assert "| 0.85 | 5 | 0 | 0.5 | 5 | 0.0183 | 71.2% | 31.0% | 15.5% | 5.10 |" in rendered
    assert "3,994 stays, 568 positive" in rendered
    assert "1,000 resamples in two strata; 400 datasets" in rendered


def test_the_decomposition_table_sets_the_measured_coverage_beside_the_predicted_one() -> None:
    parts = [
        Decomposition(
            level=0.7, repeats=5, spread=0.01, datasets=300, over_stays=0.0055, over_both=0.0084
        ),
        Decomposition(
            level=0.85, repeats=5, spread=0.01, datasets=300, over_stays=0.0042, over_both=0.0076
        ),
    ]
    measured = [
        RankingCalibration(
            level=0.7,
            gain=0.02,
            repeats=5,
            spread=0.01,
            shared=0.5,
            levels=0,
            units=3994,
            positives=568,
            datasets=400,
            resamples=1000,
            coverage=0.798,
            false_positive=0.2,
            one_sided_false_positive=0.1,
            seconds=6.0,
        )
    ]

    rendered = render_decomposition(parts, measured)

    assert "| 0.70 | 5 | 0.01 | 0.0055 | 0.0084 | 0.0063 (0.0063) | 80.1% | 79.8% |" in rendered
    assert "| 0.85 | 5 | 0.01 | 0.0042 | 0.0076 |" in rendered
    assert rendered.rstrip().endswith("| — |")


def test_the_report_runs_end_to_end_and_renders_again_from_the_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scripts.ranking_calibration_report.UNITS", 200)
    monkeypatch.setattr("scripts.ranking_calibration_report.POSITIVES", 28)
    main(["--out", str(tmp_path / "out"), "--datasets", "2", "--resamples", "10"])
    main(["--out", str(tmp_path / "out"), "--decompose", "--datasets", "3"])
    both = capsys.readouterr().out.split("## ")[-1]
    main(["--report-only", str(tmp_path / "out")])
    again = capsys.readouterr().out.split("## ")[-1]

    assert both == again
    assert "| 0.70 | 1 | 0 | 0.5 | — |" in both
    assert "| 0.85 | 5 | 0.03 |" in both
    assert "Coverage predicted" in both


def test_the_robustness_suite_runs_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scripts.ranking_calibration_report.UNITS", 200)
    monkeypatch.setattr("scripts.ranking_calibration_report.POSITIVES", 28)
    main(["--out", str(tmp_path), "--suite", "robustness", "--datasets", "2", "--resamples", "10"])

    rendered = capsys.readouterr().out
    assert "| 0.60 | 5 | 0 | 0.5 | — |" in rendered
    assert "| 0.80 | 5 | 0 | 0.5 | 5 |" in rendered
    assert "Coverage predicted" not in rendered


def test_a_run_without_a_direction_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--out DIR or --report-only DIR"):
        main([])
    with pytest.raises(SystemExit, match="nothing to render"):
        main(["--report-only", str(tmp_path)])


def test_a_run_over_no_process_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--workers must be at least one, got 0"):
        main(["--out", str(tmp_path), "--workers", "0"])
