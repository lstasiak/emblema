"""Cells of two campaigns set side by side, paired as their purpose says; nothing is trained."""

import csv
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest

from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.evaluation.domain.campaign.campaign_reading import CampaignReading
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from scripts.campaign_pairs_report import (
    CELL_COLUMNS,
    CELLS,
    COMPARISON_COLUMNS,
    COMPARISONS,
    PREDICTIONS,
    Answers,
    Comparison,
    Side,
    export,
    export_predictions,
    main,
    pair,
    read_cells,
    read_predictions,
    render,
    write_cells,
    write_predictions,
)
from tests.evaluation.support import (
    CONTENDER,
    CONTROL,
    answered,
    candidate,
    design,
    reading,
    result,
    store,
)

FLOOR = CampaignId(UUID(int=20))
DOUBLED = CampaignId(UUID(int=40))
BUDGET = LabelBudget.of(200)
BOOTSTRAP = PairedUnitBootstrap(resamples=400, seed=1, level=0.95)


def selection(
    campaign_id: CampaignId, errors: dict[CandidateRef, tuple[float, ...]]
) -> CampaignReading:
    """A finished selection of the control and the contender at 200, three repeats each."""
    grid = reading(
        campaign_id=campaign_id,
        purpose=RunPurpose.SELECTION,
        design=design(
            candidates=(candidate(CONTROL), candidate(CONTENDER)),
            budgets=(BUDGET,),
            seeds=(1, 2, 3),
            inner_holdout=InnerHoldout(one_in=5),
        ),
    )
    for cell in grid.campaign.design.cells():
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


def comparison(campaign_id: CampaignId, shift: float) -> CampaignReading:
    """A comparison whose grid ran whole on the same units under every seed."""
    grid = reading(campaign_id=campaign_id, design=design(budgets=(BUDGET,), seeds=(1, 2, 3)))
    for cell in grid.campaign.design.cells():
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
        floor_area=0.01,
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
    # Seed by seed: each repeat's error on its own, the control's less the candidate's.
    each = ((36 + 64 + 100) / 3) ** 0.5 - ((25 + 49 + 81) / 3) ** 0.5
    assert read.per_seed == pytest.approx((each, each, each))


def test_the_reduction_seed_by_seed_is_stated_with_its_standard_error() -> None:
    floor = selection(FLOOR, {CONTROL: (6.0, 8.0, 10.0), CONTENDER: (6.0, 8.0, 10.0)})
    read = pair(
        export([floor]),
        side(FLOOR, CONTROL),
        side(FLOOR, CONTENDER),
        bootstrap=BOOTSTRAP,
        floor_share=0.02,
        floor_area=0.01,
    )

    spread = replace(read, per_seed=(0.01, 0.03, 0.05))

    assert spread.seed_mean == pytest.approx(0.03)
    assert spread.seed_se == pytest.approx(0.02 / 3**0.5)


def test_a_comparison_pools_its_repeats_per_unit_before_pairing() -> None:
    first = comparison(FLOOR, shift=0.0)
    second = comparison(DOUBLED, shift=-1.0)
    read = pair(
        export([first, second]),
        side(FLOOR, CONTENDER),
        side(DOUBLED, CONTENDER),
        bootstrap=BOOTSTRAP,
        floor_share=0.02,
        floor_area=0.01,
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
        floor_area=0.01,
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
            floor_area=0.01,
        )
    with pytest.raises(ValueError, match="no cell of"):
        pair(
            rows,
            side(FLOOR, CONTROL),
            Side(str(FLOOR), "lora", "200"),
            bootstrap=BOOTSTRAP,
            floor_share=0.02,
            floor_area=0.01,
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
            floor_area=0.01,
        )


def test_main_exports_from_the_registry_then_reads_the_pairs_off_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = InMemoryEvaluationCampaignRepository()
    store(registry, selection(FLOOR, {CONTROL: (6.0, 8.0, 10.0), CONTENDER: (6.0, 8.0, 10.0)}))
    store(registry, selection(DOUBLED, {CONTROL: (5.0, 7.0, 9.0), CONTENDER: (6.0, 8.0, 10.0)}))
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
        floor_area=0.01,
    )
    rendered = render([read])
    assert "unit-pooled, 3 repeats, 3 pairs" in rendered
    assert f"{read.floor.value:.3f}" in rendered


def test_main_refuses_a_command_line_that_asks_for_nothing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path)])


BY_AREA = CampaignId(UUID(int=60))
BY_AREA_TOO = CampaignId(UUID(int=80))
# Stays s0 and s2 died. The control ranks one dead stay below two survivors, an area of 6/8.
CONTROL_ANSWERS = (0.6, 0.4, 0.3, 0.5, 0.2, 0.1)
# Every dead stay above every survivor: an area of one.
PERFECT_ANSWERS = (0.9, 0.2, 0.8, 0.3, 0.1, 0.05)


def by_area(
    campaign_id: CampaignId,
    contender: tuple[float, ...],
    purpose: RunPurpose = RunPurpose.TUNING,
) -> CampaignReading:
    """A campaign read by area whose grid ran whole on the same six stays under every seed."""
    grid = reading(
        campaign_id=campaign_id,
        purpose=purpose,
        design=replace(
            design(
                budgets=(BUDGET,),
                seeds=(1, 2, 3),
                inner_holdout=InnerHoldout(one_in=5) if purpose is RunPurpose.SELECTION else None,
            ),
            measure=ErrorMeasure.AUROC_SHORTFALL,
        ),
    )
    for cell in grid.campaign.design.cells():
        answers = CONTROL_ANSWERS if cell.candidate == CONTROL else contender
        # Each seed moves every answer alike, which leaves each repeat's ranking as it is.
        grid = grid.record(
            answered(
                cell.candidate, cell.budget, cell.seed, [a * (1 + cell.seed / 100) for a in answers]
            )
        )
    return grid


def paired_by_area(rows_of: list[CampaignReading], control: Side, contender: Side) -> Comparison:
    return pair(
        export(rows_of),
        control,
        contender,
        bootstrap=BOOTSTRAP,
        floor_share=0.02,
        floor_area=0.01,
        answers=Answers(export_predictions(rows_of)),
    )


def test_a_comparison_by_area_pools_the_areas_of_its_repeats_and_states_its_floor_in_area() -> None:
    read = paired_by_area(
        [by_area(BY_AREA, PERFECT_ANSWERS)], side(BY_AREA, CONTROL), side(BY_AREA, CONTENDER)
    )

    assert read.measure is ErrorMeasure.AUROC_SHORTFALL
    assert read.pairing == "area-pooled"
    assert (read.repeats, read.units) == (3, 6)
    assert read.control_error.pooled == pytest.approx(0.25)
    assert read.control_error.spread == 0.0
    assert read.candidate_error.pooled == 0.0
    assert read.difference.reduction == pytest.approx(0.25)
    assert read.difference.relative_reduction == pytest.approx(1.0)
    assert read.floor.value == pytest.approx(0.01)


def test_two_campaigns_read_by_area_pair_on_the_stays_both_answered() -> None:
    first, second = by_area(BY_AREA, CONTROL_ANSWERS), by_area(BY_AREA_TOO, PERFECT_ANSWERS)

    read = paired_by_area([first, second], side(BY_AREA, CONTENDER), side(BY_AREA_TOO, CONTENDER))

    assert read.difference.reduction == pytest.approx(0.25)
    named = dict(zip(COMPARISON_COLUMNS, read.row(), strict=True))
    assert named["measure"] == "auroc_shortfall"
    # Every repeat ranks the stays as every other does, so each seed gains what the pool gains.
    assert read.per_seed == pytest.approx((0.25, 0.25, 0.25))
    assert float(named["seed_reduction_mean"]) == pytest.approx(0.25)
    assert float(named["seed_reduction_se"]) == pytest.approx(0.0)


def test_the_answers_of_a_campaign_read_by_area_are_written_and_read_back(tmp_path: Path) -> None:
    answers = export_predictions([by_area(BY_AREA, PERFECT_ANSWERS), comparison(FLOOR, 0.0)])

    assert len(answers) == 2 * 3 * 6
    assert {row.campaign for row in answers} == {str(BY_AREA)}
    write_predictions(tmp_path / PREDICTIONS, answers)
    assert read_predictions(tmp_path / PREDICTIONS) == answers
    assert read_predictions(tmp_path / "absent.csv") == []


def test_a_file_of_cells_written_before_the_measure_column_reads_as_errors(tmp_path: Path) -> None:
    rows = export([comparison(FLOOR, shift=0.0)])
    write_cells(tmp_path / CELLS, rows)
    with (tmp_path / CELLS).open(newline="") as handle:
        written = list(csv.DictReader(handle))
    older = [column for column in CELL_COLUMNS if column != "measure"]
    with (tmp_path / CELLS).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=older, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(written)

    assert read_cells(tmp_path / CELLS) == rows


def test_an_export_of_errors_alone_leaves_no_answers_of_an_earlier_export_behind(
    tmp_path: Path,
) -> None:
    registry = InMemoryEvaluationCampaignRepository()
    store(registry, by_area(BY_AREA, PERFECT_ANSWERS))
    store(registry, comparison(FLOOR, shift=0.0))
    main(["--out", str(tmp_path), "--campaign", str(BY_AREA)], registry=registry)
    main(["--out", str(tmp_path), "--campaign", str(FLOOR)], registry=registry)

    assert read_predictions(tmp_path / PREDICTIONS) == []


def test_a_selection_whose_repeats_scored_the_same_units_pairs_by_area() -> None:
    # Divided under a fixed seed, every repeat scores the same fifth, as a comparison's do.
    selected = by_area(BY_AREA, PERFECT_ANSWERS, purpose=RunPurpose.SELECTION)

    read = paired_by_area([selected], side(BY_AREA, CONTROL), side(BY_AREA, CONTENDER))

    assert read.pairing == "area-pooled"
    assert read.measure is ErrorMeasure.AUROC_SHORTFALL
    assert read == paired_by_area(
        [by_area(BY_AREA, PERFECT_ANSWERS)], side(BY_AREA, CONTROL), side(BY_AREA, CONTENDER)
    )


def test_a_selection_whose_repeats_scored_other_units_is_refused_as_a_pair_by_area() -> None:
    selected = by_area(BY_AREA, PERFECT_ANSWERS, purpose=RunPurpose.SELECTION)
    rows = export([selected])
    moved = [
        row._replace(unit=f"other-{row.unit}") if row.seed == 2 else row
        for row in export_predictions([selected])
    ]

    with pytest.raises(ValueError, match="pairs only where every repeat scored the same units"):
        pair(
            rows,
            side(BY_AREA, CONTROL),
            side(BY_AREA, CONTENDER),
            bootstrap=BOOTSTRAP,
            floor_share=0.02,
            floor_area=0.01,
            answers=Answers(moved),
        )


def test_sides_read_by_different_measures_are_refused() -> None:
    with pytest.raises(ValueError, match="different measures"):
        paired_by_area(
            [comparison(FLOOR, 0.0), by_area(BY_AREA, PERFECT_ANSWERS)],
            side(FLOOR, CONTROL),
            side(BY_AREA, CONTROL),
        )


def test_a_side_whose_answers_were_not_exported_is_refused() -> None:
    grid = by_area(BY_AREA, PERFECT_ANSWERS)

    with pytest.raises(ValueError, match="no answer of"):
        pair(
            export([grid]),
            side(BY_AREA, CONTROL),
            side(BY_AREA, CONTENDER),
            bootstrap=BOOTSTRAP,
            floor_share=0.02,
            floor_area=0.01,
        )


def test_render_names_the_measure_and_refuses_to_mix_two() -> None:
    area = paired_by_area(
        [by_area(BY_AREA, PERFECT_ANSWERS)], side(BY_AREA, CONTROL), side(BY_AREA, CONTENDER)
    )
    errors = pair(
        export([comparison(FLOOR, 0.0)]),
        side(FLOOR, CONTROL),
        side(FLOOR, CONTENDER),
        bootstrap=BOOTSTRAP,
        floor_share=0.02,
        floor_area=0.01,
    )

    assert "| control | 1 - AUROC | candidate | 1 - AUROC |" in render([area])
    assert "area-pooled, 3 repeats, 6 pairs" in render([area])
    with pytest.raises(ValueError, match="one table holds one measure"):
        render([area, errors])


def test_main_exports_the_answers_of_campaigns_read_by_area_and_pairs_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = InMemoryEvaluationCampaignRepository()
    store(registry, by_area(BY_AREA, CONTROL_ANSWERS))
    store(registry, by_area(BY_AREA_TOO, PERFECT_ANSWERS))
    main(
        [
            "--out",
            str(tmp_path),
            "--campaign",
            str(BY_AREA),
            "--campaign",
            str(BY_AREA_TOO),
            "--pair",
            *side(BY_AREA, CONTENDER),
            *side(BY_AREA_TOO, CONTENDER),
            "--resamples",
            "200",
            "--floor-area",
            "0.02",
        ],
        registry=registry,
    )

    printed = capsys.readouterr().out
    assert "72 answers of the campaigns read by area" in printed
    assert "| control | 1 - AUROC |" in printed
    header = (tmp_path / COMPARISONS).read_text().splitlines()[0].split(",")
    assert header[8:12] == [
        "error_control",
        "spread_control",
        "error_candidate",
        "spread_candidate",
    ]
    written = dict(
        zip(header, (tmp_path / COMPARISONS).read_text().splitlines()[1].split(","), strict=True)
    )
    assert written["measure"] == "auroc_shortfall"
    assert float(written["floor"]) == pytest.approx(0.02)
    assert "per seed, mean ± SE" in printed
