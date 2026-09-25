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

from emblema.evaluation.adapters.candidates.backbone_arm_catalogue import BackboneArmCatalogue
from emblema.evaluation.adapters.candidates.classical_arm import ClassicalArm
from emblema.evaluation.adapters.candidates.classical_baseline_catalogue import (
    ClassicalBaselineCatalogue,
)
from emblema.evaluation.adapters.candidates.patch_model_catalogue import PatchModelCatalogue
from emblema.evaluation.adapters.candidates.routed_candidate_catalogue import (
    RoutedCandidateCatalogue,
)
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.classical.boosted_trees import BoostedTrees
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.evaluation.domain.exceptions import UnknownCandidateError
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.evaluation.ports.candidate_catalogue import CandidateCatalogue
from tests.evaluation.support import (
    CONTENDER,
    LORA,
    WEIGHTS,
    adaptation_schedule,
    arm,
    boosting,
    design,
    patch_spec,
)

TREES = CandidateRef("boosted_trees_per_channel")
ARMS = BackboneArmCatalogue(
    (arm(CONTENDER, TransferMode.FULL_FINE_TUNING, backbone=WEIGHTS, lora=LORA),)
)
BASELINES = ClassicalBaselineCatalogue(
    (
        ClassicalArm(
            ref=TREES,
            method=BoostedTrees(features=FeatureScheme.PER_CHANNEL, boosting=boosting()),
            sources=(),
        ),
    )
)
PATCHES = CandidateRef("patch_transformer")
PATCHED = PatchModelCatalogue(PATCHES, patch_spec(), adaptation_schedule())
ROUTED = RoutedCandidateCatalogue({CONTENDER: ARMS, TREES: BASELINES, PATCHES: PATCHED})
CATALOGUES: dict[str, Callable[[], tuple[CandidateCatalogue, CandidateRef]]] = {
    "arms": lambda: (ARMS, CONTENDER),
    "baselines": lambda: (BASELINES, TREES),
    "patch model": lambda: (PATCHED, PATCHES),
    "routed": lambda: (ROUTED, TREES),
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


def test_a_patch_model_starts_from_nothing_and_shares_the_arms_budget() -> None:
    described = PATCHED.describe(PATCHES)

    assert described.kind is CandidateKind.NEURAL
    assert described.starts_from is None
    # Derived from one schedule, so it is the budget the arms declare and not a copy of it.
    assert described.budget == ARMS.describe(CONTENDER).budget


def test_a_patch_model_records_its_shape_and_the_schedule_it_learns_under() -> None:
    stated = {p.name: p.value for p in PATCHED.describe(PATCHES).method.parameters}

    assert stated["patch_length"] == str(patch_spec().patch_length)
    assert stated["width"] == str(patch_spec().width)
    assert stated["learning_rate"] == str(adaptation_schedule().learning_rate)


def test_a_design_holds_the_arms_the_patch_model_and_a_baseline_side_by_side() -> None:
    # Two networks and a fit of trees in one grid: the networks' budgets are equal because both
    # are read off one schedule, and the baseline declares none.
    stated = design(
        candidates=(
            ROUTED.describe(CONTENDER),
            ROUTED.describe(PATCHES),
            ROUTED.describe(TREES),
        ),
        control=CONTENDER,
        endpoint=PATCHES,
    )

    assert {c.ref for c in stated.candidates} == {CONTENDER, PATCHES, TREES}


def test_a_variant_of_the_patch_model_turns_the_schedule_and_keeps_the_shape_and_budget() -> None:
    base = PATCHED.describe(PATCHES)

    variant = PATCHED.describe(CandidateRef("patch_transformer@learning_rate=0.003"))

    stated = {p.name: p.value for p in variant.method.parameters}
    before = {p.name: p.value for p in base.method.parameters}
    assert {name for name in stated if stated[name] != before[name]} == {"learning_rate"}
    assert variant.budget == base.budget
    assert PATCHED.plan_of(variant.ref, seed=1).schedule.learning_rate == 0.003


def test_a_knob_of_the_patch_models_shape_is_not_turned_by_a_name() -> None:
    with pytest.raises(UnknownCandidateError, match="no knob"):
        PATCHED.describe(CandidateRef("patch_transformer@patch_length=8"))


def test_a_variant_of_an_arm_is_the_arm_under_a_turned_schedule_on_the_same_budget() -> None:
    base = ARMS.describe(CONTENDER)

    variant = ARMS.describe(CandidateRef("full_fine_tuning@learning_rate=0.003,weight_decay=0.1"))

    stated = {p.name: p.value for p in variant.method.parameters}
    before = {p.name: p.value for p in base.method.parameters}
    assert {name for name in stated if stated[name] != before[name]} == {
        "learning_rate",
        "weight_decay",
    }
    assert (variant.budget, variant.starts_from, variant.kind) == (
        base.budget,
        base.starts_from,
        base.kind,
    )
    turned = ARMS.arm_of(variant.ref)
    assert (turned.schedule.learning_rate, turned.schedule.weight_decay) == (0.003, 0.1)
    assert ARMS.arm_of(CONTENDER).schedule == adaptation_schedule()


def test_the_arm_that_starts_from_nothing_names_the_model_it_is_shaped_like() -> None:
    control = BackboneArmCatalogue(
        (arm(CandidateRef("from_scratch"), TransferMode.FROM_SCRATCH, backbone=None, lora=None),)
    ).describe(CandidateRef("from_scratch"))

    stated = {p.name: p.value for p in control.method.parameters}
    assert control.starts_from is None
    assert stated["architecture_of"] == f"{WEIGHTS.key}@{WEIGHTS.checksum}"
    assert "architecture_of" not in {p.name for p in ARMS.describe(CONTENDER).method.parameters}


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


def test_a_variant_of_a_baseline_is_described_with_its_knob_turned_and_nothing_else() -> None:
    base = BASELINES.describe(TREES)

    variant = ROUTED.describe(CandidateRef("boosted_trees_per_channel@max_depth=5"))

    stated = {p.name: p.value for p in variant.method.parameters}
    before = {p.name: p.value for p in base.method.parameters}
    assert variant.ref == CandidateRef("boosted_trees_per_channel@max_depth=5")
    assert stated["max_depth"] == "5"
    assert {name for name in stated if stated[name] != before[name]} == {"max_depth"}


@pytest.mark.parametrize(
    "name",
    [
        "boosted_trees_per_channel@threads=8",
        "boosted_trees_per_channel@max_depth=deep",
        "boosted_trees_per_channel@max_depth=5,learning_rate=0.1",
        "full_fine_tuning@epochs=3",
        "full_fine_tuning@learning_rate=fast",
        "boosted_trees_per_channel@max_depth=3",
        "full_fine_tuning@learning_rate=0.01",
        "patch_transformer@learning_rate=0.01",
    ],
    ids=[
        "not a knob",
        "not its type",
        "out of order",
        "a knob that would change the budget",
        "not the schedule's type",
        "the default under another name",
        "the arm's default under another name",
        "the patch model's default under another name",
    ],
)
def test_a_variant_no_holder_can_read_is_refused(name: str) -> None:
    with pytest.raises(UnknownCandidateError):
        ROUTED.describe(CandidateRef(name))
