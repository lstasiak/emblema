"""Cells of two campaigns set side by side, paired as their purpose says; nothing is trained."""

from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest

from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from scripts.campaign_pairs_report import (
    CELLS,
    COMPARISONS,
    Side,
    export,
    main,
    pair,
    read_cells,
    render,
    write_cells,
)
from tests.evaluation.support import (
    CONTENDER,
    CONTROL,
    campaign,
    candidate,
    design,
    result,
)

FLOOR = CampaignId(UUID(int=20))
DOUBLED = CampaignId(UUID(int=40))
BUDGET = LabelBudget.of(200)
BOOTSTRAP = PairedUnitBootstrap(resamples=400, seed=1, level=0.95)


def selection(
    campaign_id: CampaignId, errors: dict[CandidateRef, tuple[float, ...]]
) -> EvaluationCampaign:
    """A finished selection of the control and the contender at 200, three repeats each."""
    grid = campaign(
        campaign_id=campaign_id,
        purpose=RunPurpose.SELECTION,
        design=design(
            candidates=(candidate(CONTROL), candidate(CONTENDER)),
            budgets=(BUDGET,),
            seeds=(1, 2, 3),
            inner_holdout=InnerHoldout(one_in=5),
        ),
    )
    for cell in grid.design.cells():
        # Every repeat of a selection holds out units of its own.
        stated = result(cell.candidate, cell.budget, cell.seed, errors[cell.candidate])
        grid = grid.record(
            replace(
                stated,
                errors=tuple(
                    replace(error, unit=replace(error.unit, value=f"s{cell.seed}-{error.unit}"))
                    for error in stated.errors
                ),
            )
        )
    return grid


def comparison(campaign_id: CampaignId, shift: float) -> EvaluationCampaign:
    """A comparison whose grid ran whole on the same units under every seed."""
    grid = campaign(campaign_id=campaign_id, design=design(budgets=(BUDGET,), seeds=(1, 2, 3)))
    for cell in grid.design.cells():
        errors = (6.0, 8.0, 10.0) if cell.candidate == CONTROL else (3.0, 4.0, 5.0)
        grid = grid.record(
            result(cell.candidate, cell.budget, cell.seed, tuple(e + shift for e in errors))
        )
    return grid


def side(campaign_id: CampaignId, ref: CandidateRef) -> Side:
    return Side(str(campaign_id), str(ref), "200")


def test_export_writes_one_row_per_unit_of_every_recorded_cell_and_reads_back(
    tmp_path: Path,
) -> None:
    floor = selection(FLOOR, {CONTROL: (6.0, 8.0, 10.0), CONTENDER: (5.0, 7.0, 9.0)})
    rows = export([floor])
    assert len(rows) == 2 * 3 * 3
    assert {row.purpose for row in rows} == {"selection"}
    assert {row.tier for row in rows} == {"S"}
    first = rows[0]
    assert (first.campaign, first.candidate, first.budget, first.seed) == (
        str(FLOOR),
        "from_scratch",
        "200",
        1,
    )
    assert first.unit == "s1-c0"
    assert first.squared_error == 36.0
    write_cells(tmp_path / CELLS, rows)
    assert read_cells(tmp_path / CELLS) == rows


def test_a_selection_pairs_every_repeat_and_unit_on_its_own() -> None:
    floor = selection(FLOOR, {CONTROL: (6.0, 8.0, 10.0), CONTENDER: (6.0, 8.0, 10.0)})
    doubled = selection(DOUBLED, {CONTROL: (5.0, 7.0, 9.0), CONTENDER: (6.0, 8.0, 10.0)})
    read = pair(
        export([floor, doubled]),
        side(FLOOR, CONTROL),
        side(DOUBLED, CONTROL),
        bootstrap=BOOTSTRAP,
        floor_share=0.02,
    )
    assert read.pairing == "repeat-unit"
    assert (read.repeats, read.units) == (3, 9)
    assert read.control_error.pooled == pytest.approx(((36 + 64 + 100) / 3) ** 0.5)
    assert read.control_error.spread == 0.0
    assert read.difference.reduction == pytest.approx(
        read.control_error.pooled - read.candidate_error.pooled
    )
    assert read.difference.interval.above_zero
    assert read.floor.value == pytest.approx(0.02 * read.control_error.pooled)


def test_a_comparison_pools_its_repeats_per_unit_before_pairing() -> None:
    first = comparison(FLOOR, shift=0.0)
    second = comparison(DOUBLED, shift=-1.0)
    read = pair(
        export([first, second]),
        side(FLOOR, CONTENDER),
        side(DOUBLED, CONTENDER),
        bootstrap=BOOTSTRAP,
        floor_share=0.02,
    )
    assert read.pairing == "unit-pooled"
    assert (read.repeats, read.units) == (3, 3)
    assert read.difference.reduction == pytest.approx((50 / 3) ** 0.5 - (29 / 3) ** 0.5)


def test_a_pair_within_one_campaign_reads_a_contender_against_its_control() -> None:
    first = comparison(FLOOR, shift=0.0)
    read = pair(
        export([first]),
        side(FLOOR, CONTROL),
        side(FLOOR, CONTENDER),
        bootstrap=BOOTSTRAP,
        floor_share=0.02,
    )
    assert read.difference.relative_reduction == pytest.approx(0.5)


def test_sides_of_different_purposes_or_seeds_or_none_at_all_are_refused() -> None:
    rows = export(
        [
            selection(FLOOR, {CONTROL: (6.0, 8.0, 10.0), CONTENDER: (5.0, 7.0, 9.0)}),
            comparison(DOUBLED, shift=0.0),
        ]
    )
    with pytest.raises(ValueError, match="different purposes"):
        pair(
            rows,
            side(FLOOR, CONTROL),
            side(DOUBLED, CONTROL),
            bootstrap=BOOTSTRAP,
            floor_share=0.02,
        )
    with pytest.raises(ValueError, match="no cell of"):
        pair(
            rows,
            side(FLOOR, CONTROL),
            Side(str(FLOOR), "lora", "200"),
            bootstrap=BOOTSTRAP,
            floor_share=0.02,
        )
    short = [
        row
        for row in rows
        if not (row.campaign == str(FLOOR) and row.candidate == str(CONTROL) and row.seed == 3)
    ]
    with pytest.raises(ValueError, match="same seeds"):
        pair(
            short,
            side(FLOOR, CONTROL),
            side(FLOOR, CONTENDER),
            bootstrap=BOOTSTRAP,
            floor_share=0.02,
        )


def test_main_exports_from_the_registry_then_reads_the_pairs_off_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = InMemoryEvaluationCampaignRepository()
    registry.save(
        selection(FLOOR, {CONTROL: (6.0, 8.0, 10.0), CONTENDER: (6.0, 8.0, 10.0)}), seen=0
    )
    registry.save(
        selection(DOUBLED, {CONTROL: (5.0, 7.0, 9.0), CONTENDER: (6.0, 8.0, 10.0)}), seen=0
    )
    main(
        [
            "--out",
            str(tmp_path),
            "--campaign",
            str(FLOOR),
            "--campaign",
            str(DOUBLED),
            "--pair",
            *side(FLOOR, CONTROL),
            *side(DOUBLED, CONTROL),
            "--pair",
            *side(FLOOR, CONTENDER),
            *side(DOUBLED, CONTENDER),
            "--resamples",
            "200",
        ],
        registry=registry,
    )
    printed = capsys.readouterr().out
    assert "36 unit errors of 2 campaigns" in printed
    assert "| control | RMSE | candidate | RMSE | reduction |" in printed
    lines = (tmp_path / COMPARISONS).read_text().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("control_campaign,control,candidate_campaign")
    assert lines[2].split(",")[12] == repr(0.0)


def test_render_names_the_pairing_and_the_floor() -> None:
    first = comparison(FLOOR, shift=0.0)
    read = pair(
        export([first]),
        side(FLOOR, CONTROL),
        side(FLOOR, CONTENDER),
        bootstrap=BOOTSTRAP,
        floor_share=0.02,
    )
    rendered = render([read])
    assert "unit-pooled, 3 repeats, 3 pairs" in rendered
    assert f"{read.floor.value:.3f}" in rendered


def test_main_refuses_a_command_line_that_asks_for_nothing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path)])
