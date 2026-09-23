"""What any provider of candidates owes the port, whatever it supplies.

The contract says nothing about how a candidate is built or how well it answers: it holds every
adapter to describing what it supplies in the terms a design is stated in, to answering a cell
with an error per unit, to keeping what it fitted exactly when it was asked to, and to refusing
a candidate it does not supply. What each adapter supplies differs — a network records the
weights it starts from and shares the campaign's budget, a baseline does neither — so each one
says so beside itself and the contract asks about that rather than about a fixed answer. The
provider that adapts a backbone is exercised through the torch runtime elsewhere; here both
stand over runtimes that learn the mean, so the contract is checked without a tensor.
"""

from collections.abc import Callable
from dataclasses import replace
from typing import NamedTuple

import pytest

from emblema.evaluation.adapters.candidates.backbone_arm import BackboneArm
from emblema.evaluation.adapters.candidates.backbone_candidate_provider import (
    BackboneCandidateProvider,
)
from emblema.evaluation.adapters.candidates.classical_arm import ClassicalArm
from emblema.evaluation.adapters.candidates.classical_candidate_provider import (
    ClassicalCandidateProvider,
)
from emblema.evaluation.adapters.candidates.routed_candidate_provider import (
    RoutedCandidateProvider,
)
from emblema.evaluation.adapters.in_memory.adaptation_runtime import InMemoryAdaptationRuntime
from emblema.evaluation.adapters.in_memory.candidate_provider import InMemoryCandidateProvider
from emblema.evaluation.adapters.in_memory.classical_runtime import InMemoryClassicalRuntime
from emblema.evaluation.adapters.in_memory.corpus_windows import InMemoryCorpusWindows
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.in_memory.ground_truth import InMemoryGroundTruth
from emblema.evaluation.application.use_cases.draw_label_budget import DrawLabelBudget
from emblema.evaluation.application.use_cases.draw_run_labels import DrawRunLabels
from emblema.evaluation.application.use_cases.open_test_split import OpenTestSplit
from emblema.evaluation.application.use_cases.run_adaptation import RunAdaptation
from emblema.evaluation.application.use_cases.run_classical_fit import RunClassicalFit
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.campaign.candidate_method import CandidateMethod
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.evaluation.domain.exceptions import (
    CandidateMethodMismatchError,
    CandidateNotRetainableError,
    UnknownBackboneError,
    UnknownCandidateError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.evaluation.ports.candidate_provider import CandidateProvider
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.kernel.artifacts import ArtifactRef
from tests.evaluation.support import (
    CONTENDER,
    CONTROL,
    LORA,
    OPENED_AT,
    WEIGHTS,
    adaptation_schedule,
    boosting,
    candidate,
    cell,
    sides,
    task,
    units,
)

ENDS = {
    UnitKey("a"): [60.0, 120.0, 180.0, 240.0],
    UnitKey("b"): [60.0, 120.0, 180.0, 240.0],
    UnitKey("c"): [60.0, 120.0],
}
FAILURES = {UnitKey("a"): 300.0, UnitKey("b"): 260.0, UnitKey("c"): 200.0}
PUBLISHED = sides(training=units("a", "b"), validation=units("c"))
LOW_RANK = CandidateRef("lora")
TREES = CandidateRef("boosted_trees_per_channel")
BUDGET = LabelBudget.of(2)


class Supplied(NamedTuple):
    """An adapter, the candidate asked of it here, and what a campaign records that one as.

    A network starts from weights and shares the grid's compute budget; a baseline does
    neither. Both are true of the port, so what differs is stated beside each adapter instead
    of being asserted as if one of the two were the rule.
    """

    provider: CandidateProvider
    contender: CandidateRef
    starts_from: ArtifactRef | None
    shares_a_budget: bool


def drawing() -> tuple[InMemoryDownstreamTaskRepository, DrawRunLabels, DrawLabelBudget]:
    """One task over one corpus, and the labelling every provider here is built over."""
    tasks = InMemoryDownstreamTaskRepository()
    tasks.save(task())
    corpus = InMemoryCorpusWindows(PUBLISHED, ENDS, task().manifest)
    truth = InMemoryGroundTruth(FAILURES)
    budgets = DrawLabelBudget(tasks, corpus, truth)
    return (
        tasks,
        DrawRunLabels(
            tasks,
            corpus,
            truth,
            budgets,
            OpenTestSplit(
                tasks,
                SequentialIdGenerator(),
                FixedClock(OPENED_AT),
                InMemoryEventPublisher(InMemoryEventSubscriber()),
            ),
        ),
        budgets,
    )


def stated_provider(store: InMemoryArtifactStore | None) -> Supplied:
    return Supplied(
        provider=InMemoryCandidateProvider(
            (candidate(CONTROL), candidate(CONTENDER)),
            (UnitKey("c"),),
            lambda _: (2.0,),
            store,
        ),
        contender=CONTENDER,
        starts_from=WEIGHTS,
        shares_a_budget=True,
    )


def backbone_provider(store: InMemoryArtifactStore | None) -> Supplied:
    _, labels, _ = drawing()
    return Supplied(
        provider=BackboneCandidateProvider(
            (
                BackboneArm(ref=CONTROL, mode=TransferMode.FROM_SCRATCH, backbone=None, lora=None),
                BackboneArm(
                    ref=CONTENDER,
                    mode=TransferMode.FULL_FINE_TUNING,
                    backbone=WEIGHTS,
                    lora=None,
                ),
                BackboneArm(ref=LOW_RANK, mode=TransferMode.LORA, backbone=WEIGHTS, lora=LORA),
            ),
            adaptation_schedule(),
            RunAdaptation(labels, InMemoryAdaptationRuntime(store)),
        ),
        contender=CONTENDER,
        starts_from=WEIGHTS,
        shares_a_budget=True,
    )


def classical_provider(store: InMemoryArtifactStore | None) -> Supplied:
    tasks, labels, budgets = drawing()
    return Supplied(
        provider=ClassicalCandidateProvider(
            (ClassicalArm(ref=TREES, features=FeatureScheme.PER_CHANNEL, sources=()),),
            boosting(),
            RunClassicalFit(tasks, labels, budgets, InMemoryClassicalRuntime(store)),
        ),
        contender=TREES,
        starts_from=None,
        shares_a_budget=False,
    )


def routed_provider(store: InMemoryArtifactStore | None) -> Supplied:
    """Both kinds behind one port; the contract asks it for the one it routes second."""
    return Supplied(
        provider=RoutedCandidateProvider(
            {
                CONTROL: backbone_provider(store).provider,
                CONTENDER: backbone_provider(store).provider,
                TREES: classical_provider(store).provider,
            }
        ),
        contender=TREES,
        starts_from=None,
        shares_a_budget=False,
    )


ADAPTERS: dict[str, Callable[[InMemoryArtifactStore | None], Supplied]] = {
    "stated": stated_provider,
    "backbone": backbone_provider,
    "classical": classical_provider,
    "routed": routed_provider,
}


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def supplied(request: pytest.FixtureRequest) -> Supplied:
    build: Callable[[InMemoryArtifactStore | None], Supplied] = request.param
    return build(None)


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def keeping(request: pytest.FixtureRequest) -> Supplied:
    build: Callable[[InMemoryArtifactStore | None], Supplied] = request.param
    return build(InMemoryArtifactStore())


def request_of(supplied: Supplied, retain: bool = False) -> CandidateEvaluation:
    """A cell as a campaign designed over this very provider would have recorded it."""
    return CandidateEvaluation(
        task=task().task_id,
        cell=cell(supplied.contender, BUDGET, 1),
        purpose=RunPurpose.TUNING,
        retain=retain,
        starts_from=supplied.starts_from,
        method=supplied.provider.describe(supplied.contender).method,
    )


def test_a_candidate_is_described_in_the_terms_a_design_is_stated_in(supplied: Supplied) -> None:
    described = supplied.provider.describe(supplied.contender)

    assert described.ref == supplied.contender
    assert (described.budget is not None) == supplied.shares_a_budget
    assert described.starts_from == supplied.starts_from


def test_a_candidate_the_provider_does_not_supply_is_refused(supplied: Supplied) -> None:
    with pytest.raises(UnknownCandidateError):
        supplied.provider.describe(CandidateRef("absent"))


def test_a_cell_is_answered_with_an_error_per_unit(supplied: Supplied) -> None:
    result = supplied.provider.evaluate(request_of(supplied))

    assert result.cell == cell(supplied.contender, BUDGET, 1)
    assert [error.unit for error in result.errors] == [UnitKey("c")]
    assert result.artifact is None


def test_a_cell_the_provider_was_asked_to_keep_names_what_it_stored(keeping: Supplied) -> None:
    result = keeping.provider.evaluate(request_of(keeping, retain=True))

    assert result.artifact is not None


def test_a_provider_with_nowhere_to_keep_what_it_fits_refuses_to_keep_it(
    supplied: Supplied,
) -> None:
    with pytest.raises(CandidateNotRetainableError):
        supplied.provider.evaluate(request_of(supplied, retain=True))


def stated_method(candidate: CandidateRef) -> dict[str, str]:
    described = backbone_provider(None).provider.describe(candidate)
    return {parameter.name: parameter.value for parameter in described.method.parameters}


def test_an_arm_records_the_schedule_it_learns_under_beyond_the_arithmetic_it_may_spend() -> None:
    stated = stated_method(CONTENDER)

    assert stated["transfer_mode"] == "full_fine_tuning"
    assert stated["learning_rate"] == "0.01"
    assert stated["weight_decay"] == "0.0"
    assert "epochs" not in stated


def test_a_low_rank_arm_records_which_layers_it_updates_and_how_strongly() -> None:
    stated = stated_method(LOW_RANK)

    assert stated["lora_rank"] == "2"
    assert stated["lora_targets"] == "qkv, attention.projection, feedforward"


def test_the_backbone_adapter_refuses_a_cell_run_over_other_weights() -> None:
    other = ArtifactRef(key="durable/other", checksum=WEIGHTS.checksum)
    asked = replace(request_of(backbone_provider(None)), starts_from=other)

    with pytest.raises(UnknownBackboneError, match="other weights"):
        backbone_provider(None).provider.evaluate(asked)


def test_a_baseline_records_how_it_reads_a_window_and_how_hard_it_fits() -> None:
    described = classical_provider(None).provider.describe(TREES)
    stated = {parameter.name: parameter.value for parameter in described.method.parameters}

    assert stated["features"] == "per_channel"
    assert stated["rounds"] == "8"
    # What a fit answers depends on it, so a campaign that recorded every other knob and
    # left this one would record a method that does not identify its own result.
    assert stated["threads"] == "1"
    assert stated["sources"] == ""
    assert "fit_seed" not in stated


def test_the_classical_adapter_refuses_a_cell_the_campaign_recorded_as_starting_from_weights() -> (
    None
):
    supplied = classical_provider(None)
    asked = replace(request_of(supplied), starts_from=WEIGHTS)

    with pytest.raises(UnknownBackboneError, match="starting from weights"):
        supplied.provider.evaluate(asked)


def test_each_name_reaches_the_supplier_the_process_named_for_it() -> None:
    routed = routed_provider(None).provider

    assert routed.describe(CONTENDER).kind is CandidateKind.NEURAL
    assert routed.describe(TREES).kind is CandidateKind.CLASSICAL


def test_a_name_no_supplier_was_named_for_is_refused_before_anyone_is_asked() -> None:
    with pytest.raises(UnknownCandidateError, match="this process supplies no candidate"):
        routed_provider(None).provider.describe(CandidateRef("lora"))


def test_a_cell_recorded_under_other_settings_than_the_provider_holds_is_refused(
    supplied: Supplied,
) -> None:
    # A worker configured otherwise would answer a point of the curve under settings the grid
    # never declared, and the grid would read as though one candidate had been compared at one
    # setting. The comparison is of the text a provider states, never of what it means.
    elsewhere = replace(
        request_of(supplied), method=CandidateMethod.of(learning_rate=0.5, weight_decay=0.5)
    )

    with pytest.raises(CandidateMethodMismatchError):
        supplied.provider.evaluate(elsewhere)
