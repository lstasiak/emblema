"""What any catalogue of candidates owes the port, whatever it holds.

Declaring a comparison and running one are two things, and this is the first: a catalogue says
what a candidate is, refuses a name it does not hold, and needs nothing that could produce the
answer. That last part is the reason the port exists, so it is asserted rather than assumed —
a process that only declares a campaign must carry neither stack the grid will be run with.
"""

import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace

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
from emblema.evaluation.domain.exceptions import InvalidBackboneArmError, UnknownCandidateError
from emblema.evaluation.domain.heads.head_pooling import (
    HeadPooling,
    PoolingScheme,
    StaticsPlacement,
)
from emblema.evaluation.domain.transfer.encoder_setting import EncoderSetting, ValueEmbedding
from emblema.evaluation.domain.transfer.encoder_shape import EncoderShape
from emblema.evaluation.domain.transfer.layer_reading import LayerReading
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.evaluation.ports.candidate_catalogue import CandidateCatalogue
from tests.evaluation.support import (
    CONTENDER,
    LORA,
    PENALTIES,
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
BOTH_ARMS = BackboneArmCatalogue(
    (
        arm(CONTENDER, TransferMode.FULL_FINE_TUNING, backbone=WEIGHTS, lora=LORA),
        arm(CandidateRef("from_scratch"), TransferMode.FROM_SCRATCH, backbone=None, lora=None),
    )
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


def test_a_knob_of_the_patch_models_shape_is_turned_by_a_name_on_the_same_budget() -> None:
    base = PATCHED.describe(PATCHES)

    variant = PATCHED.describe(CandidateRef("patch_transformer@layers=2"))

    stated = {p.name: p.value for p in variant.method.parameters}
    assert stated["layers"] == "2"
    assert variant.budget == base.budget
    assert PATCHED.plan_of(variant.ref, seed=1).spec.layers == 2


def test_a_knob_no_part_of_the_patch_model_has_is_refused() -> None:
    with pytest.raises(UnknownCandidateError, match="no knob"):
        PATCHED.describe(CandidateRef("patch_transformer@depth=8"))


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


def test_an_arm_whose_mode_solves_its_head_names_penalties() -> None:
    solved = arm(
        CandidateRef("frozen_ridge"), TransferMode.FROZEN_RIDGE, backbone=WEIGHTS, lora=None
    )
    with pytest.raises(InvalidBackboneArmError, match="names no penalties"):
        replace(solved, ridge=None)
    described = BackboneArmCatalogue((solved,)).describe(solved.ref)
    stated = {parameter.name: parameter.value for parameter in described.method.parameters}
    assert stated["ridge_penalties"] == "0.1 1 10"
    assert described.budget == ARMS.describe(CONTENDER).budget


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


def test_a_variant_of_an_arm_turns_the_pooling_of_its_head_on_the_same_budget() -> None:
    base = ARMS.describe(CONTENDER)

    variant = ARMS.describe(CandidateRef("full_fine_tuning@pooling=tail,tail_share=0.2"))

    stated = {p.name: p.value for p in variant.method.parameters}
    assert (stated["pooling"], stated["tail_share"]) == ("tail", "0.2")
    assert variant.budget == base.budget
    turned = ARMS.arm_of(variant.ref)
    assert turned.pooling == HeadPooling(pooling=PoolingScheme.TAIL, tail_share=0.2)
    assert turned.schedule == adaptation_schedule()


def test_a_variant_of_the_patch_model_turns_the_pooling_of_its_head() -> None:
    variant = PATCHED.describe(CandidateRef("patch_transformer@pooling=attention"))

    stated = {p.name: p.value for p in variant.method.parameters}
    assert stated["pooling"] == "attention"
    assert PATCHED.plan_of(variant.ref, seed=1).pooling == HeadPooling(
        pooling=PoolingScheme.ATTENTION
    )


@pytest.mark.parametrize(
    "name",
    [
        # A share means nothing to the mean.
        "full_fine_tuning@tail_share=0.2",
        "patch_transformer@pooling=median",
        "full_fine_tuning@statics=beside",
        # The grid holds a static feature as a channel, so the patch model has none to set apart.
        "patch_transformer@statics=apart",
    ],
)
def test_a_pooling_no_head_can_take_is_refused(name: str) -> None:
    with pytest.raises(UnknownCandidateError, match="names no variant"):
        ROUTED.describe(CandidateRef(name))


def test_an_arm_at_the_standard_encoder_is_described_as_before_the_encoder_had_knobs() -> None:
    # A campaign checks each cell's candidate against the description it stored, whole: one
    # more column here would refuse every cell and selection of the campaigns that ran before.
    described = ARMS.describe(CONTENDER)

    assert [p.name for p in described.method.parameters] == sorted(
        [
            "transfer_mode",
            "learning_rate",
            "weight_decay",
            "warmup_fraction",
            "final_lr_fraction",
            "pooling",
            "tail_share",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_targets",
        ]
    )


def test_a_variant_of_an_arm_turns_its_encoder_on_the_same_budget() -> None:
    base = ARMS.describe(CONTENDER)

    variant = ARMS.describe(CandidateRef("full_fine_tuning@dropout=0.2,grid_resolution=1"))

    stated = {p.name: p.value for p in variant.method.parameters}
    before = {p.name: p.value for p in base.method.parameters}
    assert {name: stated[name] for name in stated.keys() - before.keys()} == {
        "encoder_dropout": "0.2",
        "grid_resolution": "1.0",
    }
    assert all(stated[name] == before[name] for name in before)
    assert variant.budget == base.budget
    turned = ARMS.arm_of(variant.ref)
    assert turned.encoder == EncoderSetting(dropout=0.2, grid_resolution=1.0)
    assert turned.schedule == adaptation_schedule()


def test_a_variant_that_sets_static_features_apart_names_only_that_on_the_same_budget() -> None:
    base = ARMS.describe(CONTENDER)

    variant = ARMS.describe(CandidateRef("full_fine_tuning@pooling=attention,statics=apart"))

    stated = {p.name: p.value for p in variant.method.parameters}
    before = {p.name: p.value for p in base.method.parameters}
    assert {name: stated[name] for name in stated.keys() - before.keys()} == {"statics": "apart"}
    assert stated["pooling"] == "attention"
    assert variant.budget == base.budget
    assert ARMS.arm_of(variant.ref).pooling == HeadPooling(
        pooling=PoolingScheme.ATTENTION, statics=StaticsPlacement.APART
    )


def test_a_dropout_is_refused_where_the_probe_states_every_window_once() -> None:
    probes = BackboneArmCatalogue(
        (arm(CandidateRef("frozen_probe"), TransferMode.FROZEN_PROBE, backbone=WEIGHTS, lora=None),)
    )

    with pytest.raises(UnknownCandidateError, match="a dropout would change nothing"):
        probes.describe(CandidateRef("frozen_probe@dropout=0.2"))
    # Knobs turn in name order, the dropout's before the pooling's: the arm is judged once all
    # of them are turned, so the pooling that puts the encoder in the loop is counted.
    learnt = probes.arm_of(CandidateRef("frozen_probe@dropout=0.2,pooling=attention"))
    assert learnt.encoder.dropout == 0.2
    assert learnt.pooling == HeadPooling(pooling=PoolingScheme.ATTENTION)


@pytest.mark.parametrize(
    "name",
    [
        "full_fine_tuning@dropout=1",
        "full_fine_tuning@grid_resolution=0",
        # Pretrained weights fix the encoder's build.
        "full_fine_tuning@value_embedding=nonlinear",
        "full_fine_tuning@feedforward_width=128,heads=16,layers=2,width=64",
        # A shape stated in part, or whose width its heads cannot split.
        "from_scratch@heads=16,width=64",
        "from_scratch@feedforward_width=128,heads=5,layers=2,width=64",
        # A stop stated in part.
        "from_scratch@patience=10",
    ],
)
def test_an_encoder_setting_no_encoder_can_take_is_refused(name: str) -> None:
    with pytest.raises(UnknownCandidateError, match="names no variant"):
        BOTH_ARMS.describe(CandidateRef(name))


def test_a_probe_reads_another_layer_and_is_described_by_that_alone() -> None:
    solved = arm(
        CandidateRef("frozen_ridge"), TransferMode.FROZEN_RIDGE, backbone=WEIGHTS, lora=None
    )
    ridges = BackboneArmCatalogue((solved,))
    base = ridges.describe(solved.ref)

    variant = ridges.describe(CandidateRef("frozen_ridge@layer=concat,pooling=tail"))

    stated = {p.name: p.value for p in variant.method.parameters}
    before = {p.name: p.value for p in base.method.parameters}
    assert {name: stated[name] for name in stated.keys() - before.keys()} == {
        "encoder_layer": "concat"
    }
    assert variant.budget == base.budget
    assert ridges.arm_of(variant.ref).encoder.layer == LayerReading.of("concat")


@pytest.mark.parametrize("name", ["full_fine_tuning@layer=2", "from_scratch@layer=mean"])
def test_a_layer_below_the_last_is_refused_where_the_run_trains_the_encoder(name: str) -> None:
    with pytest.raises(UnknownCandidateError, match="trains the encoder"):
        BOTH_ARMS.describe(CandidateRef(name))


def test_a_network_from_nothing_names_its_own_build_and_nothing_else() -> None:
    base = BOTH_ARMS.describe(CandidateRef("from_scratch"))

    variant = BOTH_ARMS.describe(
        CandidateRef(
            "from_scratch@feedforward_width=128,heads=16,layers=2,value_embedding=nonlinear,width=64"
        )
    )

    stated = {p.name: p.value for p in variant.method.parameters}
    before = {p.name: p.value for p in base.method.parameters}
    assert {name: stated[name] for name in stated.keys() - before.keys()} == {
        "value_embedding": "nonlinear",
        "encoder_width": "64",
        "encoder_heads": "16",
        "encoder_layers": "2",
        "encoder_feedforward_width": "128",
    }
    assert variant.budget == base.budget
    turned = BOTH_ARMS.arm_of(variant.ref).encoder
    assert turned.shape == EncoderShape(width=64, heads=16, layers=2, feedforward_width=128)
    assert turned.value_embedding is ValueEmbedding.NONLINEAR


def test_a_variant_of_an_arm_turns_its_regime_and_is_described_by_the_knobs_turned() -> None:
    base = BOTH_ARMS.describe(CandidateRef("from_scratch"))

    variant = BOTH_ARMS.describe(
        CandidateRef(
            "from_scratch@channel_dropout=0.2,class_weight=ratio,patience=10,stop_share=0.2"
        )
    )

    stated = {p.name: p.value for p in variant.method.parameters}
    before = {p.name: p.value for p in base.method.parameters}
    assert {name: stated[name] for name in stated.keys() - before.keys()} == {
        "stop_share": "0.2",
        "patience": "10",
        "class_weight": "ratio",
        "channel_dropout": "0.2",
    }
    assert variant.budget == base.budget
    turned = BOTH_ARMS.arm_of(variant.ref).regime
    assert turned.stops
    assert turned.channel_dropout == 0.2


def test_a_regime_is_refused_where_the_probe_states_every_window_once() -> None:
    probes = BackboneArmCatalogue(
        (arm(CandidateRef("frozen_probe"), TransferMode.FROZEN_PROBE, backbone=WEIGHTS, lora=None),)
    )

    with pytest.raises(UnknownCandidateError, match="withholding channels would change nothing"):
        probes.describe(CandidateRef("frozen_probe@channel_dropout=0.2"))
    # Under a learnt pooling the encoder is in the loop, and the channels can be withheld.
    learnt = probes.arm_of(CandidateRef("frozen_probe@channel_dropout=0.2,pooling=attention"))
    assert learnt.regime.channel_dropout == 0.2


HOLDING = BackboneArmCatalogue(
    (
        arm(CONTENDER, TransferMode.FULL_FINE_TUNING, backbone=WEIGHTS, lora=None, ridge=PENALTIES),
        arm(CandidateRef("from_scratch"), TransferMode.FROM_SCRATCH, backbone=None, lora=None),
    )
)


def test_penalties_an_arm_holds_for_a_head_solved_first_leave_its_description_as_it_was() -> None:
    # A campaign stored before the head could start solved checks its cells against this
    # description, whole, so penalties held for a variant must not reach the base arm's.
    held = {p.name: p.value for p in HOLDING.describe(CONTENDER).method.parameters}
    plain = {
        p.name: p.value
        for p in BackboneArmCatalogue(
            (arm(CONTENDER, TransferMode.FULL_FINE_TUNING, backbone=WEIGHTS, lora=None),)
        )
        .describe(CONTENDER)
        .method.parameters
    }

    assert held == plain
    assert HOLDING.arm_of(CONTENDER).solved_under is None


def test_a_variant_solves_the_head_first_and_is_described_by_its_penalties_and_the_knob() -> None:
    base = {p.name: p.value for p in HOLDING.describe(CONTENDER).method.parameters}
    variant = HOLDING.describe(CandidateRef("full_fine_tuning@head_start=solved"))

    stated = {p.name: p.value for p in variant.method.parameters}
    assert {name: stated[name] for name in stated.keys() - base.keys()} == {
        "head_start": "solved",
        "ridge_penalties": "0.1 1 10",
    }
    assert variant.budget == HOLDING.describe(CONTENDER).budget
    assert HOLDING.arm_of(variant.ref).solved_under == PENALTIES


def test_a_head_solved_first_is_refused_without_penalties_or_under_a_learnt_pooling() -> None:
    with pytest.raises(UnknownCandidateError, match="no penalties to solve its head among"):
        BOTH_ARMS.describe(CandidateRef("from_scratch@head_start=solved"))
    with pytest.raises(UnknownCandidateError, match="cannot learn a attention pooling"):
        HOLDING.describe(CandidateRef("full_fine_tuning@head_start=solved,pooling=attention"))


def test_a_stop_in_steps_divided_by_outcome_is_a_variant_and_a_division_alone_is_refused() -> None:
    variant = HOLDING.describe(
        CandidateRef("from_scratch@patience_steps=40,stop_division=outcomes,stop_share=0.2")
    )

    stated = {p.name: p.value for p in variant.method.parameters}
    assert stated["patience_steps"] == "40"
    assert stated["stop_division"] == "outcomes"
    assert "patience" not in stated
    with pytest.raises(UnknownCandidateError, match="a division only with both"):
        HOLDING.describe(CandidateRef("from_scratch@stop_division=outcomes"))


def test_a_probe_over_the_encoder_at_its_initialisation_names_the_model_it_is_shaped_like() -> None:
    untrained = BackboneArmCatalogue(
        (
            arm(
                CandidateRef("untrained_ridge"),
                TransferMode.FROZEN_RIDGE,
                backbone=None,
                lora=None,
            ),
        )
    ).describe(CandidateRef("untrained_ridge"))

    stated = {p.name: p.value for p in untrained.method.parameters}
    assert untrained.starts_from is None
    assert stated["architecture_of"] == f"{WEIGHTS.key}@{WEIGHTS.checksum}"
    assert stated["ridge_penalties"] == "0.1 1 10"
