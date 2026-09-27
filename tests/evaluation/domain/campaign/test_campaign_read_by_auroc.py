"""A campaign over a binary task: read by how its candidates rank the outcomes apart."""

from dataclasses import replace

import pytest

from emblema.evaluation.domain.campaign.campaign_reading import CampaignReading
from emblema.evaluation.domain.exceptions import PredictionsNotKeptError
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from emblema.evaluation.domain.statistics.comparison_verdict import ComparisonVerdict
from emblema.evaluation.domain.statistics.threshold_kind import ThresholdKind
from tests.evaluation.support import (
    CLOSED_AT,
    CONTENDER,
    CONTROL,
    FINER,
    ROCKET,
    RULES,
    answered,
    design,
    reading,
    result,
    selection,
)

AT_200 = LabelBudget.of(200)
# Stays 0 and 2 are the positives: the control puts one of them under two negatives, the
# contender ranks every positive above every negative.
CONTROL_ANSWERS = (0.6, 0.5, 0.2, 0.3, 0.1, 0.4)
CONTENDER_ANSWERS = (0.9, 0.2, 0.8, 0.3, 0.1, 0.4)
# A control that ranks every positive under every negative: any resample of the stays holds a
# positive and a negative, and the contender gains the whole area in every one of them.
BACKWARDS = (0.1, 0.5, 0.2, 0.6, 0.7, 0.8)


def by_auroc(answers: dict[str, tuple[float, ...]] | None = None) -> CampaignReading:
    stated = answers or {"control": CONTROL_ANSWERS, "contender": CONTENDER_ANSWERS}
    rules = replace(
        RULES, threshold=ThresholdKind.ABSOLUTE, minimum_reduction=0.05, floor_part=0.01
    )
    grid = reading(design=replace(design(), measure=ErrorMeasure.AUROC_SHORTFALL, rules=rules))
    for run in grid.campaign.design.cells():
        side = "control" if run.candidate == CONTROL else "contender"
        grid = grid.record(answered(run.candidate, run.budget, run.seed, stated[side]))
    return grid.complete(CLOSED_AT)


def test_a_candidate_is_scored_by_one_minus_its_area_under_the_roc_curve() -> None:
    # Of the eight positive-negative pairs the control orders five correctly.
    assert by_auroc().error_of(CONTROL, AT_200) == pytest.approx(1.0 - 5.0 / 8.0)
    assert by_auroc().error_of(CONTENDER, AT_200) == 0.0


def test_the_endpoint_reads_the_gain_in_area_against_an_absolute_minimum() -> None:
    endpoint = by_auroc({"control": BACKWARDS, "contender": CONTENDER_ANSWERS}).verdict().endpoint

    assert endpoint.difference.reduction == 1.0
    assert (endpoint.difference.interval.low, endpoint.difference.interval.high) == (1.0, 1.0)
    assert endpoint.verdict is ComparisonVerdict.CONFIRMED


def test_a_gain_some_resamples_of_the_stays_do_not_hold_is_not_confirmed() -> None:
    # The control misorders one positive; a resample that draws only the other positive sees no
    # difference, so the interval reaches zero.
    endpoint = by_auroc().verdict().endpoint

    assert endpoint.difference.reduction == pytest.approx(3.0 / 8.0)
    assert endpoint.difference.interval.low == 0.0
    assert endpoint.verdict is ComparisonVerdict.INDISTINGUISHABLE


def test_the_sentence_states_the_area_raised_rather_than_an_error_lowered() -> None:
    sentence = by_auroc().verdict().sentence()

    assert "full_fine_tuning raises the AUROC of from_scratch by 0.375 (0.625 → 1," in sentence
    assert "closing 100.0% of its shortfall" in sentence


def test_a_control_that_ranks_perfectly_is_read_without_a_share_of_its_shortfall() -> None:
    verdict = by_auroc({"control": CONTENDER_ANSWERS, "contender": CONTENDER_ANSWERS}).verdict()

    assert verdict.endpoint.difference.relative_reduction is None
    assert verdict.endpoint.verdict is ComparisonVerdict.INDISTINGUISHABLE
    assert "(1 → 1, the control ranking perfectly, with 95% interval" in verdict.sentence()


def test_a_candidate_that_ranks_no_better_is_indistinguishable() -> None:
    same = by_auroc({"control": CONTROL_ANSWERS, "contender": CONTROL_ANSWERS})

    assert same.verdict().endpoint.verdict is ComparisonVerdict.INDISTINGUISHABLE


def test_a_campaign_read_by_area_refuses_cells_recorded_without_their_answers() -> None:
    grid = reading(design=replace(design(), measure=ErrorMeasure.AUROC_SHORTFALL))
    for run in grid.campaign.design.cells():
        grid = grid.record(result(run.candidate, run.budget, run.seed, (1.0, 2.0)))

    with pytest.raises(PredictionsNotKeptError, match="without its answers"):
        grid.complete(CLOSED_AT).verdict()


def test_a_selection_read_by_area_chooses_the_variant_that_ranks_better() -> None:
    chosen = selection()
    by_area = replace(
        chosen,
        campaign=replace(
            chosen.campaign,
            design=replace(chosen.campaign.design, measure=ErrorMeasure.AUROC_SHORTFALL),
        ),
    )
    outcomes = (1.0, 0.0, 1.0, 0.0)
    better = (0.9, 0.1, 0.8, 0.2)
    worse = (0.1, 0.9, 0.8, 0.2)
    results = tuple(
        answered(
            run.cell.candidate,
            run.cell.budget,
            run.cell.seed,
            better if run.cell.candidate == FINER else worse,
            outcomes,
        )
        for run in chosen.results
    )

    assert replace(by_area, results=results).selected(ROCKET, AT_200) == FINER
