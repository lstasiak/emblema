"""A finished comparison read out of the registry into the files a note is made from."""

from dataclasses import replace
from pathlib import Path

import pytest

from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.domain.campaign.campaign_reading import CampaignReading
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from scripts.campaign_report import (
    CELLS,
    COMPARISONS,
    VERDICT,
    export_cells,
    export_comparisons,
    main,
    render,
    windows_of,
)
from tests.evaluation.support import (
    CAMPAIGN,
    CLOSED_AT,
    CONTENDER,
    CONTROL,
    SELECTED_BY,
    answered,
    closed_reading,
    design,
    ran_reading,
    reading,
    selection,
    store,
)


def test_every_recorded_cell_becomes_a_row_in_the_grids_order() -> None:
    cells = export_cells(closed_reading(), everything=None)

    assert len(cells) == 8
    assert [(c.candidate, c.budget, c.seed) for c in cells[:3]] == [
        ("from_scratch", "50", 1),
        ("from_scratch", "50", 2),
        ("from_scratch", "200", 1),
    ]
    assert cells[0].kind == "neural"
    assert cells[0].units == 3
    assert cells[0].score == pytest.approx(((36 + 64 + 100) / 3) ** 0.5)
    assert cells[0].brier is None


def test_every_comparison_becomes_a_row_the_endpoint_first() -> None:
    campaign = closed_reading()

    rows = export_comparisons(campaign.verdict(), campaign, everything=None)

    assert [(r.candidate, r.budget, r.primary) for r in rows] == [
        ("full_fine_tuning", "200", "yes"),
        ("full_fine_tuning", "50", "no"),
    ]
    assert rows[0].verdict == "confirmed"
    assert rows[0].reduction == pytest.approx(rows[0].control_score - rows[0].candidate_score)
    assert rows[0].low <= rows[0].reduction <= rows[0].high


def test_the_budget_of_every_window_needs_its_count() -> None:
    assert windows_of(LabelBudget.of(50), None) == 50
    assert windows_of(LabelBudget.everything(), 2568) == 2568
    with pytest.raises(SystemExit, match="--everything"):
        windows_of(LabelBudget.everything(), None)


def test_the_rendering_holds_the_sentence_and_a_table_per_budget() -> None:
    campaign = closed_reading()
    verdict = campaign.verdict()
    cells = export_cells(campaign, None)

    rendered = render(campaign, verdict, cells, export_comparisons(verdict, campaign, None))

    assert rendered.startswith(verdict.sentence())
    assert rendered.count("labelled windows**") == 2
    assert "| from_scratch |" in rendered
    assert "| control |" in rendered
    assert "confirmed (endpoint)" in rendered


def test_main_writes_the_three_files_and_prints_the_sentence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = InMemoryEvaluationCampaignRepository()
    store(registry, closed_reading())

    main(["--campaign", str(CAMPAIGN), "--out", str(tmp_path)], registry=registry)

    assert (tmp_path / CELLS).read_text().splitlines()[0] == (
        "candidate,kind,budget,windows,seed,units,rmse,seconds"
    )
    assert len((tmp_path / COMPARISONS).read_text().splitlines()) == 3
    assert (tmp_path / VERDICT).read_text().startswith(closed_reading().verdict().sentence())
    assert closed_reading().verdict().sentence() in capsys.readouterr().out


def test_a_selection_or_an_unfinished_campaign_is_refused(tmp_path: Path) -> None:
    registry = InMemoryEvaluationCampaignRepository()
    store(registry, selection())
    partial = reading()
    for produced in ran_reading().results[:-1]:
        partial = partial.record(produced)
    store(registry, partial)

    with pytest.raises(SystemExit, match="selection"):
        main(["--campaign", str(SELECTED_BY), "--out", str(tmp_path)], registry=registry)
    with pytest.raises(SystemExit, match="finished"):
        main(["--campaign", str(CAMPAIGN), "--out", str(tmp_path)], registry=registry)


def binary_campaign() -> CampaignReading:
    grid = reading(design=replace(design(), measure=ErrorMeasure.AUROC_SHORTFALL))
    answers = {CONTROL: (0.1, 0.5, 0.2, 0.6, 0.7, 0.8), CONTENDER: (0.9, 0.2, 0.8, 0.3, 0.1, 0.4)}
    for run in grid.campaign.design.cells():
        grid = grid.record(answered(run.candidate, run.budget, run.seed, answers[run.candidate]))
    return grid.complete(CLOSED_AT)


def test_a_campaign_read_by_area_reports_areas_and_brier_scores(tmp_path: Path) -> None:
    registry = InMemoryEvaluationCampaignRepository()
    store(registry, binary_campaign())

    main(["--campaign", str(CAMPAIGN), "--out", str(tmp_path)], registry=registry)

    cells = (tmp_path / CELLS).read_text().splitlines()
    comparisons = (tmp_path / COMPARISONS).read_text().splitlines()
    assert cells[0] == "candidate,kind,budget,windows,seed,units,auroc,brier,seconds"
    assert cells[1].startswith("from_scratch,neural,50,50,1,6,0.0,")
    assert comparisons[0].startswith("candidate,budget,windows,repeats,control_auroc,")
    assert comparisons[1].startswith("full_fine_tuning,200,200,2,0.0,0.0,1.0,0.0,1.0,1.0,")
    rendered = (tmp_path / VERDICT).read_text()
    assert "AUROC mean ± SD" in rendered
    assert "| full_fine_tuning | 1.00 ± 0.00 | +1.000 |" in rendered
