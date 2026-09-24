"""A design stored and read back, since a campaign resumed from the database is read from this."""

from dataclasses import replace

from emblema.evaluation.adapters.persistence.campaign_design_document import (
    CampaignDesignDocument,
)
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.candidate_method import CandidateMethod
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.tuning.tuned_choice import TunedChoice
from tests.evaluation.support import (
    CONTENDER,
    CONTROL,
    FINER,
    PROBE,
    ROCKET,
    SELECTED_BY,
    baseline,
    candidate,
    design,
    selection,
)

DOCUMENTS = CampaignDesignDocument()


def test_a_design_comes_back_as_it_was_written() -> None:
    stated = design()

    assert DOCUMENTS.decode(DOCUMENTS.encode(stated)) == stated


def test_the_budget_of_every_label_survives_the_round_trip() -> None:
    stated = design(
        budgets=(LabelBudget.of(50), LabelBudget.everything()),
        endpoint_budget=LabelBudget.everything(),
    )

    assert DOCUMENTS.decode(DOCUMENTS.encode(stated)) == stated


def test_what_each_candidate_was_configured_with_survives_the_round_trip() -> None:
    stated = design(
        candidates=(
            candidate(CONTROL),
            candidate(
                CONTENDER, method=CandidateMethod.of(learning_rate=1e-3, lora_targets="qkv, ffn")
            ),
        )
    )

    assert DOCUMENTS.decode(DOCUMENTS.encode(stated)) == stated


def test_a_candidate_whose_provider_declared_no_parameter_comes_back_declaring_none() -> None:
    stated = design(candidates=(candidate(CONTROL, method=CandidateMethod()), candidate(CONTENDER)))

    assert DOCUMENTS.decode(DOCUMENTS.encode(stated)) == stated


def test_a_classical_candidate_comes_back_holding_no_shared_budget() -> None:
    stated = design(
        candidates=(
            candidate(CONTROL),
            candidate(CONTENDER),
            candidate(PROBE, kind=CandidateKind.CLASSICAL, budget=None, starts_from=None),
        )
    )

    assert DOCUMENTS.decode(DOCUMENTS.encode(stated)) == stated


def test_the_weights_a_candidate_starts_from_survive_the_round_trip() -> None:
    stated = design()

    read = DOCUMENTS.decode(DOCUMENTS.encode(stated))

    assert read.get_candidate(CONTENDER).starts_from == stated.get_candidate(CONTENDER).starts_from


def test_the_registered_rules_and_the_bootstrap_survive_the_round_trip() -> None:
    stated = design(bootstrap=replace(design().bootstrap, resamples=17, seed=9, level=0.9))

    read = DOCUMENTS.decode(DOCUMENTS.encode(stated))

    assert read.bootstrap == stated.bootstrap
    assert read.rules == stated.rules


def test_a_selection_and_a_comparison_of_tuned_variants_survive_the_round_trip() -> None:
    chosen = selection().design
    tuned = replace(
        design(),
        candidates=(baseline(ROCKET), baseline(CandidateRef("other"))),
        control=CandidateRef("other"),
        endpoint=ROCKET,
        tuned=(
            TunedChoice(
                candidate=ROCKET, budget=LabelBudget.of(200), variant=FINER, selected_by=SELECTED_BY
            ),
        ),
        variants=(baseline(FINER, 2.0),),
    )

    assert DOCUMENTS.decode(DOCUMENTS.encode(chosen)) == chosen
    assert DOCUMENTS.decode(DOCUMENTS.encode(tuned)) == tuned


def test_a_design_stored_before_selections_existed_reads_back_tuning_nothing() -> None:
    written = DOCUMENTS.encode(design())
    for key in ("inner_holdout", "tuned", "variants"):
        del written[key]

    assert DOCUMENTS.decode(written) == design()
