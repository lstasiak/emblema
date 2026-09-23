"""What any provider of candidates owes the port, whatever it supplies.

The contract says nothing about how a candidate is built or how well it answers: it holds every
adapter to describing what it supplies in the terms a design is stated in, to answering a cell
with an error per unit, to keeping what it fitted exactly when it was asked to, and to refusing
a candidate it does not supply. The provider that adapts a backbone is exercised through the
torch runtime elsewhere; here it stands over a runtime that learns the mean, so the contract is
checked without a tensor.
"""

from collections.abc import Callable

import pytest

from emblema.evaluation.adapters.candidates.backbone_arm import BackboneArm
from emblema.evaluation.adapters.candidates.backbone_candidate_provider import (
    BackboneCandidateProvider,
)
from emblema.evaluation.adapters.in_memory.adaptation_runtime import InMemoryAdaptationRuntime
from emblema.evaluation.adapters.in_memory.candidate_provider import InMemoryCandidateProvider
from emblema.evaluation.adapters.in_memory.corpus_windows import InMemoryCorpusWindows
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.in_memory.ground_truth import InMemoryGroundTruth
from emblema.evaluation.application.use_cases.draw_label_budget import DrawLabelBudget
from emblema.evaluation.application.use_cases.draw_run_labels import DrawRunLabels
from emblema.evaluation.application.use_cases.open_test_split import OpenTestSplit
from emblema.evaluation.application.use_cases.run_adaptation import RunAdaptation
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.exceptions import (
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
CELL = cell(CONTENDER, LabelBudget.of(2), 1)
LOW_RANK = CandidateRef("lora")


def stated_provider(store: InMemoryArtifactStore | None) -> CandidateProvider:
    return InMemoryCandidateProvider(
        (candidate(CONTROL), candidate(CONTENDER)),
        (UnitKey("c"),),
        lambda _: (2.0,),
        store,
    )


def backbone_provider(store: InMemoryArtifactStore | None) -> CandidateProvider:
    tasks = InMemoryDownstreamTaskRepository()
    tasks.save(task())
    corpus = InMemoryCorpusWindows(PUBLISHED, ENDS, task().manifest)
    truth = InMemoryGroundTruth(FAILURES)
    return BackboneCandidateProvider(
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
        RunAdaptation(
            DrawRunLabels(
                tasks,
                corpus,
                truth,
                DrawLabelBudget(tasks, corpus, truth),
                OpenTestSplit(
                    tasks,
                    SequentialIdGenerator(),
                    FixedClock(OPENED_AT),
                    InMemoryEventPublisher(InMemoryEventSubscriber()),
                ),
            ),
            InMemoryAdaptationRuntime(store),
        ),
    )


ADAPTERS: dict[str, Callable[[InMemoryArtifactStore | None], CandidateProvider]] = {
    "stated": stated_provider,
    "backbone": backbone_provider,
}


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def provider(request: pytest.FixtureRequest) -> CandidateProvider:
    build: Callable[[InMemoryArtifactStore | None], CandidateProvider] = request.param
    return build(None)


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def keeping(request: pytest.FixtureRequest) -> CandidateProvider:
    build: Callable[[InMemoryArtifactStore | None], CandidateProvider] = request.param
    return build(InMemoryArtifactStore())


def request_for(retain: bool = False, starts_from: ArtifactRef | None = WEIGHTS):
    return CandidateEvaluation(
        task=task().task_id,
        cell=CELL,
        purpose=RunPurpose.TUNING,
        retain=retain,
        starts_from=starts_from,
    )


def test_a_candidate_is_described_in_the_terms_a_design_is_stated_in(
    provider: CandidateProvider,
) -> None:
    described = provider.describe(CONTENDER)

    assert described.ref == CONTENDER
    assert described.budget is not None
    assert described.starts_from == WEIGHTS


def test_a_candidate_the_provider_does_not_supply_is_refused(
    provider: CandidateProvider,
) -> None:
    with pytest.raises(UnknownCandidateError):
        provider.describe(CandidateRef("absent"))


def test_a_cell_is_answered_with_an_error_per_unit(provider: CandidateProvider) -> None:
    result = provider.evaluate(request_for())

    assert result.cell == CELL
    assert [error.unit for error in result.errors] == [UnitKey("c")]
    assert result.artifact is None


def test_a_cell_the_provider_was_asked_to_keep_names_what_it_stored(
    keeping: CandidateProvider,
) -> None:
    result = keeping.evaluate(request_for(retain=True))

    assert result.artifact is not None


def test_a_provider_with_nowhere_to_keep_what_it_fits_refuses_to_keep_it(
    provider: CandidateProvider,
) -> None:
    with pytest.raises(CandidateNotRetainableError):
        provider.evaluate(request_for(retain=True))


def stated_method(candidate: CandidateRef) -> dict[str, str]:
    described = backbone_provider(None).describe(candidate)
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

    with pytest.raises(UnknownBackboneError, match="other weights"):
        backbone_provider(None).evaluate(request_for(starts_from=other))
