"""What any catalogue of candidates owes the port, whatever it holds.

Declaring a comparison and running one are two things, and this is the first: a catalogue says
what a candidate is, refuses a name it does not hold, and needs nothing that could produce the
answer. That last part is the reason the port exists, so it is asserted rather than assumed —
a process that only declares a campaign must carry neither stack the grid will be run with.
"""

import subprocess
import sys
from collections.abc import Callable

import pytest

from emblema.evaluation.adapters.candidates.backbone_arm import BackboneArm
from emblema.evaluation.adapters.candidates.backbone_arm_catalogue import BackboneArmCatalogue
from emblema.evaluation.adapters.candidates.classical_arm import ClassicalArm
from emblema.evaluation.adapters.candidates.classical_baseline_catalogue import (
    ClassicalBaselineCatalogue,
)
from emblema.evaluation.adapters.candidates.routed_candidate_catalogue import (
    RoutedCandidateCatalogue,
)
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.evaluation.domain.exceptions import UnknownCandidateError
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.evaluation.ports.candidate_catalogue import CandidateCatalogue
from tests.evaluation.support import CONTENDER, LORA, WEIGHTS, adaptation_schedule, boosting

TREES = CandidateRef("boosted_trees_per_channel")
ARMS = BackboneArmCatalogue(
    (BackboneArm(ref=CONTENDER, mode=TransferMode.FULL_FINE_TUNING, backbone=WEIGHTS, lora=LORA),),
    adaptation_schedule(),
)
BASELINES = ClassicalBaselineCatalogue(
    (ClassicalArm(ref=TREES, features=FeatureScheme.PER_CHANNEL, sources=()),), boosting()
)
CATALOGUES: dict[str, Callable[[], tuple[CandidateCatalogue, CandidateRef]]] = {
    "arms": lambda: (ARMS, CONTENDER),
    "baselines": lambda: (BASELINES, TREES),
    "routed": lambda: (
        RoutedCandidateCatalogue({CONTENDER: ARMS, TREES: BASELINES}),
        TREES,
    ),
}


@pytest.fixture(params=list(CATALOGUES.values()), ids=list(CATALOGUES))
def held(request: pytest.FixtureRequest) -> tuple[CandidateCatalogue, CandidateRef]:
    build: Callable[[], tuple[CandidateCatalogue, CandidateRef]] = request.param
    return build()


def test_a_candidate_is_described_under_the_name_it_was_asked_for(
    held: tuple[CandidateCatalogue, CandidateRef],
) -> None:
    catalogue, candidate = held

    assert catalogue.describe(candidate).ref == candidate


def test_a_candidate_the_catalogue_does_not_hold_is_refused(
    held: tuple[CandidateCatalogue, CandidateRef],
) -> None:
    catalogue, _ = held

    with pytest.raises(UnknownCandidateError):
        catalogue.describe(CandidateRef("absent"))


def test_a_network_records_the_weights_it_starts_from_and_shares_the_grids_budget() -> None:
    described = ARMS.describe(CONTENDER)

    assert described.kind is CandidateKind.NEURAL
    assert described.starts_from == WEIGHTS
    assert described.budget is not None


def test_a_baseline_starts_from_nothing_and_shares_no_budget() -> None:
    described = BASELINES.describe(TREES)

    assert described.kind is CandidateKind.CLASSICAL
    assert described.starts_from is None
    assert described.budget is None


def test_each_name_reaches_the_holder_the_process_named_for_it() -> None:
    routed = RoutedCandidateCatalogue({CONTENDER: ARMS, TREES: BASELINES})

    assert routed.describe(CONTENDER).kind is CandidateKind.NEURAL
    assert routed.describe(TREES).kind is CandidateKind.CLASSICAL


def test_describing_every_candidate_a_campaign_may_name_loads_no_runtime() -> None:
    # The claim the port exists for, and the only way to hold it is to look at a process that
    # did it: an import inside this one would be satisfied by whatever a test imported first.
    read = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import emblema.entrypoints.cli.campaign.composition_root  # noqa: F401\n"
            "print(sorted(m for m in sys.modules if m in {'torch', 'xgboost', 'sklearn'}))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert read.stdout.strip() == "[]"
