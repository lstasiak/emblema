"""The three-panel figure drawn from a campaign report's files; nothing is computed."""

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from scripts.campaign_curve_figures import (
    ARMS,
    STEM,
    Report,
    base_of,
    draw,
    label_heights,
    main,
    read,
)
from scripts.campaign_report import (
    CELL_COLUMNS,
    CELLS,
    COMPARISON_COLUMNS,
    COMPARISONS,
    CellRow,
    ComparisonRow,
    write_rows,
)

ERRORS = {
    "from_scratch@learning_rate=0.003,pooling=tail,tail_share=0.2": 20.0,
    "frozen_probe@pooling=tail,tail_share=0.2": 19.0,
    "frozen_ridge@pooling=tail,tail_share=0.2": 17.0,
    "lora@pooling=tail,tail_share=0.2": 16.0,
    "full_fine_tuning@pooling=tail,tail_share=0.2": 15.0,
    "boosted_trees_per_channel@max_depth=3": 14.0,
    "minirocket": 16.5,
    "patch_transformer": 18.5,
}
BUDGETS = {"50": 50, "200": 200, "all": 2568}


def report() -> Report:
    cells = tuple(
        CellRow(
            candidate=candidate,
            kind="classical" if base_of(candidate).startswith(("boosted", "mini")) else "neural",
            budget=budget,
            windows=windows,
            seed=seed,
            units=21,
            rmse=error - (3.0 if budget == "all" else 0.0) + 0.3 * seed,
            seconds=1.0,
        )
        for budget, windows in BUDGETS.items()
        for candidate, error in ERRORS.items()
        for seed in (1, 2)
    )
    control = next(iter(ERRORS))
    comparisons = tuple(
        ComparisonRow(
            candidate=candidate,
            budget=budget,
            windows=windows,
            repeats=2,
            control_rmse=20.0,
            control_sd=0.2,
            candidate_rmse=error,
            candidate_sd=0.2,
            reduction=20.0 - error,
            relative_reduction=(20.0 - error) / 20.0,
            low=20.0 - error - 1.0,
            high=20.0 - error + 1.0,
            p_value=0.01,
            floor=0.4,
            primary="yes" if (candidate.startswith("full"), budget) == (True, "200") else "no",
            verdict="confirmed",
        )
        for budget, windows in BUDGETS.items()
        for candidate, error in ERRORS.items()
        if candidate != control
    )
    return Report(cells=cells, comparisons=comparisons)


def write(stored: Report, directory: Path) -> Path:
    directory.mkdir(parents=True)
    write_rows(directory / CELLS, CELL_COLUMNS, [tuple(r) for r in stored.cells])
    write_rows(directory / COMPARISONS, COMPARISON_COLUMNS, [tuple(r) for r in stored.comparisons])
    return directory


def test_the_figure_is_drawn_from_the_report(tmp_path: Path) -> None:
    figure = draw(report(), tmp_path / "figures" / "curve.png")

    assert figure.is_file()
    assert figure.stat().st_size > 10_000


def test_a_partial_grid_is_drawn_over_the_budgets_each_candidate_has(tmp_path: Path) -> None:
    whole = report()
    partial = Report(
        cells=tuple(
            c for c in whole.cells if not (c.candidate.startswith("lora") and c.budget == "all")
        ),
        comparisons=tuple(
            r
            for r in whole.comparisons
            if not (r.candidate.startswith("lora") and r.budget == "all")
        ),
    )

    assert draw(partial, tmp_path / "partial.png").is_file()


def test_the_files_round_trip_and_the_figure_lands_beside_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stored = report()
    write(stored, tmp_path / "report")

    assert read(tmp_path / "report") == stored
    main([str(tmp_path / "report")])
    assert (tmp_path / "report" / f"{STEM}.png").is_file()
    assert capsys.readouterr().out.strip().endswith(f"{STEM}.png")


def test_a_directory_without_a_report_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="nothing to draw"):
        read(tmp_path)


def test_a_variant_is_drawn_under_its_bare_name() -> None:
    assert base_of("lora@learning_rate=0.0003,pooling=tail,tail_share=0.2") == "lora"
    assert base_of("minirocket") == "minirocket"


def test_a_campaign_without_baselines_is_drawn_on_two_panels(tmp_path: Path) -> None:
    whole = report()
    networks = Report(
        cells=tuple(c for c in whole.cells if c.kind == "neural"),
        comparisons=tuple(r for r in whole.comparisons if base_of(r.candidate) in ARMS),
    )

    figure = draw(networks, tmp_path / "networks.png")

    assert figure.is_file()
    assert figure.stat().st_size < draw(whole, tmp_path / "whole.png").stat().st_size


def test_a_report_without_a_cell_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no cell"):
        draw(Report(cells=(), comparisons=()), tmp_path / "empty.png")


def test_labels_of_lines_that_end_close_together_are_set_apart_in_order() -> None:
    heights = label_heights([12.9, 15.7, 12.8, 12.3], gap=1.0)

    assert heights == pytest.approx([14.3, 15.7, 13.3, 12.3])


def test_labels_of_lines_that_end_apart_stay_at_their_ends() -> None:
    assert label_heights([30.0, 10.0, 20.0], gap=1.0) == [30.0, 10.0, 20.0]
