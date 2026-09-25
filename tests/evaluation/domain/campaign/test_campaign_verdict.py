"""The verdict as a sentence: the endpoint, the shape of the curve around it, and the side."""

import pytest

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_verdict import CampaignVerdict
from emblema.evaluation.domain.campaign.candidate_comparison import CandidateComparison
from emblema.evaluation.domain.exceptions import (
    InvalidCampaignVerdictError,
    SelectionHasNoVerdictError,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.statistics.bootstrap_interval import BootstrapInterval
from emblema.evaluation.domain.statistics.comparison_verdict import ComparisonVerdict
from emblema.evaluation.domain.statistics.error_over_repeats import ErrorOverRepeats
from emblema.evaluation.domain.statistics.paired_difference import PairedDifference
from emblema.evaluation.domain.statistics.practical_floor import PracticalFloor
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from tests.evaluation.support import CONTENDER, CONTROL, PROBE

AT_200 = LabelBudget.of(200)


def comparison(
    candidate: CandidateRef,
    budget: LabelBudget,
    verdict: ComparisonVerdict,
    *,
    reduction: float = 2.18,
    repeats: int = 5,
) -> CandidateComparison:
    control = ErrorOverRepeats.of(18.86, [18.0, 19.5, 18.9, 18.2, 19.7][:repeats])
    return CandidateComparison(
        candidate=candidate,
        budget=budget,
        control_error=control,
        candidate_error=ErrorOverRepeats.of(
            18.86 - reduction, [16.3, 17.0, 16.6, 16.5, 17.1][:repeats]
        ),
        difference=PairedDifference(
            reduction=reduction,
            relative_reduction=reduction / 18.86,
            interval=BootstrapInterval(low=reduction - 0.69, high=reduction + 1.12, level=0.95),
            p_value=0.001,
        ),
        floor=PracticalFloor(value=0.57),
        verdict=verdict,
    )


def verdict(
    endpoint: ComparisonVerdict = ComparisonVerdict.CONFIRMED,
    *,
    secondary: tuple[CandidateComparison, ...] = (),
    read_on: RunPurpose = RunPurpose.TUNING,
) -> CampaignVerdict:
    return CampaignVerdict(
        control=CONTROL,
        read_on=read_on,
        endpoint=comparison(CONTENDER, AT_200, endpoint),
        secondary=secondary,
    )


def test_a_confirmed_endpoint_opens_the_sentence_with_its_size_interval_floor_and_spread() -> None:
    sentence = verdict().sentence()

    assert sentence.startswith(
        "Confirmed on the registered endpoint: at 200 labels full_fine_tuning lowers the error "
        "of from_scratch by 11.6% (18.9 → 16.7, a reduction of 2.18 with 95% interval "
        "[1.49; 3.3], floor 0.57; over 5 and 5 repeats the sides spread 0.76 and 0.34)"
    )
    assert sentence.endswith(". Validation side; preliminary.")


def test_an_endpoint_that_fell_short_says_so_and_of_what() -> None:
    sentence = verdict(ComparisonVerdict.PRACTICALLY_NIL).sentence()

    assert sentence.startswith(
        "Not confirmed on the registered endpoint (distinguishable, practically nil): "
    )


def test_the_curve_around_the_endpoint_says_where_the_advantage_holds() -> None:
    along = (
        comparison(CONTENDER, LabelBudget.of(50), ComparisonVerdict.DISTINGUISHABLE),
        comparison(CONTENDER, LabelBudget.of(1000), ComparisonVerdict.DISTINGUISHABLE),
        comparison(CONTENDER, LabelBudget.everything(), ComparisonVerdict.WORSE),
    )

    sentence = verdict(secondary=along).sentence()

    assert (
        "; among the other budgets the advantage of full_fine_tuning holds under the family's "
        "correction and above the floor at 50 and 1000, not at the full label set."
    ) in sentence


def test_an_advantage_that_holds_everywhere_or_nowhere_is_said_that_way() -> None:
    held = comparison(CONTENDER, LabelBudget.of(50), ComparisonVerdict.DISTINGUISHABLE)
    lost = comparison(CONTENDER, LabelBudget.of(50), ComparisonVerdict.INDISTINGUISHABLE)

    assert "at each of them (50)." in verdict(secondary=(held,)).sentence()
    assert "at none of them (50)." in verdict(secondary=(lost,)).sentence()


def test_the_other_candidates_are_counted_apart_from_the_endpoints_own_curve() -> None:
    others = (
        comparison(CONTENDER, LabelBudget.of(50), ComparisonVerdict.DISTINGUISHABLE),
        comparison(PROBE, LabelBudget.of(50), ComparisonVerdict.INDISTINGUISHABLE),
        comparison(PROBE, AT_200, ComparisonVerdict.DISTINGUISHABLE),
    )

    sentence = verdict(secondary=others).sentence()

    assert "at each of them (50); 1 of 2 secondary comparisons of the other candidates are" in (
        sentence
    )


def test_a_verdict_read_on_the_test_side_is_final() -> None:
    assert verdict(read_on=RunPurpose.FINAL).sentence().endswith(". Test side; final.")


def test_one_repeat_a_side_is_said_without_a_spread() -> None:
    lone = CampaignVerdict(
        control=CONTROL,
        read_on=RunPurpose.TUNING,
        endpoint=comparison(CONTENDER, AT_200, ComparisonVerdict.CONFIRMED, repeats=1),
        secondary=(),
    )

    assert "; one repeat each)" in lone.sentence()


def test_a_selection_has_no_verdict_to_state() -> None:
    with pytest.raises(SelectionHasNoVerdictError):
        verdict(read_on=RunPurpose.SELECTION)


def test_a_verdict_that_compares_the_control_with_itself_or_a_pairing_twice_is_refused() -> None:
    with pytest.raises(InvalidCampaignVerdictError, match="against itself"):
        CampaignVerdict(
            control=CONTENDER,
            read_on=RunPurpose.TUNING,
            endpoint=comparison(CONTENDER, AT_200, ComparisonVerdict.CONFIRMED),
            secondary=(),
        )
    with pytest.raises(InvalidCampaignVerdictError, match="twice"):
        verdict(secondary=(comparison(CONTENDER, AT_200, ComparisonVerdict.DISTINGUISHABLE),))
