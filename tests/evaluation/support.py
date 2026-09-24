"""One task and one plan, stated once, for every test that needs one rather than a rule.

The overrides are typed ``Any`` because each names a field of the value object it builds and
carries that field's type; the value object refuses anything else on the way in.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef, TaskId
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.campaign_design import CampaignDesign
from emblema.evaluation.domain.campaign.candidate_method import CandidateMethod
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.campaign.compute_budget import ComputeBudget
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.classical.boosted_trees import BoostedTrees
from emblema.evaluation.domain.classical.classical_method import ClassicalMethod
from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.evaluation.domain.classical.gradient_boosting_spec import GradientBoostingSpec
from emblema.evaluation.domain.classical.minirocket_spec import MiniRocketSpec
from emblema.evaluation.domain.classical.random_convolutions import RandomConvolutions
from emblema.evaluation.domain.classical.ridge_spec import RidgeSpec
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.forecast_scheme import ForecastScheme
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.labels.target_bins import TargetBins
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.unit_error import UnitError
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.statistics.comparison_rules import ComparisonRules
from emblema.evaluation.domain.statistics.holm_correction import HolmCorrection
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.task.corpus_sides import CorpusSides
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.domain.task.task_split import TaskSplit
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.compute import ComputeTier
from emblema.shared.kernel.timestamps import UtcDateTime

CORPUS = "turbofans"
MANIFEST = ArtifactRef(key="durable/manifest", checksum=Checksum.of_bytes(b"manifest"))
TASK = TaskId(UUID(int=1))
CAMPAIGN = CampaignId(UUID(int=2))
OPENED_AT = UtcDateTime(datetime(2026, 1, 1, tzinfo=UTC))
SCHEME = RemainingLifeScheme(125.0)
FORECAST = ForecastScheme("s01", 12.0)
STRATA = TargetBins(4)
TEST_SIDE = FrozenTestSplit(units=frozenset({UnitKey("held/1")}), source="turbofans/test")
WEIGHTS = ArtifactRef(key="durable/weights", checksum=Checksum.of_bytes(b"weights"))
# The three kinds of linear layer the encoder's blocks have; the attention's output projection
# is named with its parent, because the time encoding has a projection of its own.
LORA = LoraSpec(
    rank=2, alpha=4.0, dropout=0.0, targets=("qkv", "attention.projection", "feedforward")
)


def units(*names: str) -> frozenset[UnitKey]:
    return frozenset(UnitKey(name) for name in names)


def sides(training: frozenset[UnitKey], validation: frozenset[UnitKey]) -> CorpusSides:
    return CorpusSides(corpus=CORPUS, training=training, validation=validation)


def window(unit: str, position: int, ends_at: float) -> TaskWindow:
    return TaskWindow(unit=UnitKey(unit), position=position, ends_at=ends_at)


def task(
    tuning: frozenset[UnitKey] = units("a", "b"),
    validation: frozenset[UnitKey] = units("c"),
    test: FrozenTestSplit = TEST_SIDE,
    labels: RemainingLifeScheme | ForecastScheme | None = SCHEME,
    protocol: EvaluationProtocol = EvaluationProtocol.LABEL_BUDGET,
    strata: TargetBins | None = STRATA,
) -> DownstreamTask:
    return DownstreamTask(
        task_id=TASK,
        corpus=CORPUS,
        manifest=MANIFEST,
        split=TaskSplit(tuning=tuning, validation=validation, test=test),
        protocol=protocol,
        labels=labels,
        strata=strata,
    )


def labelled(unit: str, position: int, ends_at: float, target: float) -> LabelledWindow:
    return LabelledWindow(window=window(unit, position, ends_at), target=target)


def prediction(unit: str, position: int, target: float, predicted: float) -> WindowPrediction:
    return WindowPrediction(
        window=window(unit, position, float(position)), target=target, predicted=predicted
    )


def adaptation_schedule(**overrides: Any) -> AdaptationSchedule:
    stated = AdaptationSchedule(
        epochs=2,
        min_steps=0,
        batch_size=2,
        learning_rate=1e-2,
        weight_decay=0.0,
        warmup_fraction=0.0,
        final_lr_fraction=1.0,
    )
    return replace(stated, **overrides)


def boosting(**overrides: Any) -> GradientBoostingSpec:
    """Few shallow trees on one thread: enough to fit something, fast enough for a domain test."""
    stated = GradientBoostingSpec(
        rounds=8,
        max_depth=3,
        learning_rate=0.3,
        row_share=1.0,
        feature_share=1.0,
        min_leaf_weight=1.0,
        l2_penalty=1.0,
        threads=1,
    )
    return replace(stated, **overrides)


def convolutions(**overrides: Any) -> RandomConvolutions:
    """One family of kernels on one thread: MiniRocket at its smallest."""
    stated = RandomConvolutions(
        convolutions=MiniRocketSpec(features=84),
        ridge=RidgeSpec(penalties=(0.1, 1.0, 10.0), threads=1),
    )
    return replace(stated, **overrides)


def recipe(
    features: FeatureScheme = FeatureScheme.PER_CHANNEL,
    *,
    trees: GradientBoostingSpec | None = None,
    method: ClassicalMethod | None = None,
    **overrides: Any,
) -> ClassicalRecipe:
    """A recipe that holds together; anything named is replaced afterwards.

    Boosted trees under ``features`` and ``trees`` unless another ``method`` is named.
    """
    fitted = (
        BoostedTrees(features=features, boosting=boosting() if trees is None else trees)
        if method is None
        else method
    )
    stated = ClassicalRecipe(method=fitted, seed=1, sources=())
    return replace(stated, **overrides)


def plan(mode: TransferMode = TransferMode.FULL_FINE_TUNING, **overrides: Any) -> AdaptationPlan:
    """A plan of ``mode`` that holds together; anything named is replaced afterwards."""
    stated = AdaptationPlan(
        mode=mode,
        backbone=WEIGHTS if mode.starts_from_pretrained_weights else None,
        schedule=adaptation_schedule(),
        lora=LORA if mode.adds_low_rank_updates else None,
        seed=1,
    )
    return replace(stated, **overrides)


CONTROL = CandidateRef("from_scratch")
CONTENDER = CandidateRef("full_fine_tuning")
PROBE = CandidateRef("frozen_probe")
BUDGETS = (LabelBudget.of(50), LabelBudget.of(200))
RULES = ComparisonRules(
    minimum_relative_reduction=0.1,
    floor_share=0.02,
    holm=HolmCorrection(alpha=0.05),
    secondary_family_size=8,
)
# Few resamples on purpose: a whole campaign is exercised here, and an interval read off ten
# thousand of them per comparison would make the cycle slower than the thing it stands in for.
BOOTSTRAP = PairedUnitBootstrap(resamples=200, seed=1, level=0.95)
COMPUTE = ComputeBudget(epochs=2, min_steps=0, batch_size=2)
METHOD = CandidateMethod.of(learning_rate=1e-2, weight_decay=0.0)


def candidate(ref: CandidateRef, **overrides: Any) -> CampaignCandidate:
    stated = CampaignCandidate(
        ref=ref,
        kind=CandidateKind.NEURAL,
        budget=COMPUTE,
        method=METHOD,
        starts_from=None if ref == CONTROL else WEIGHTS,
    )
    return replace(stated, **overrides)


def design(**overrides: Any) -> CampaignDesign:
    stated = CampaignDesign(
        candidates=(candidate(CONTROL), candidate(CONTENDER)),
        control=CONTROL,
        endpoint=CONTENDER,
        budgets=BUDGETS,
        endpoint_budget=LabelBudget.of(200),
        seeds=(1, 2),
        rules=RULES,
        bootstrap=BOOTSTRAP,
    )
    return replace(stated, **overrides)


def cell(ref: CandidateRef, budget: LabelBudget, seed: int) -> CampaignCell:
    return CampaignCell(candidate=ref, budget=budget, seed=seed)


def result(
    ref: CandidateRef, budget: LabelBudget, seed: int, errors: Sequence[float], **overrides: Any
) -> CellResult:
    """A cell scored over as many units as ``errors`` names, one window each."""
    stated = CellResult(
        cell=cell(ref, budget, seed),
        errors=tuple(
            UnitError(unit=UnitKey(f"c{index}"), squared_error=error**2, windows=1)
            for index, error in enumerate(errors)
        ),
        seconds=0.0,
        artifact=None,
    )
    return replace(stated, **overrides)


def campaign(**overrides: Any) -> EvaluationCampaign:
    stated = EvaluationCampaign.designed(
        campaign_id=CAMPAIGN,
        task=TASK,
        purpose=RunPurpose.TUNING,
        tier=ComputeTier.S,
        design=design(),
        opened_at=OPENED_AT,
    )
    return replace(stated, **overrides)


def artifact(name: str) -> ArtifactRef:
    """A reference to bytes named after what they stand for, for a test that keeps one."""
    return ArtifactRef(key=f"durable/{name}", checksum=Checksum.of_bytes(name.encode()))


CLOSED_AT = UtcDateTime(datetime(2026, 1, 2, tzinfo=UTC))


def ran_campaign(
    kept: ArtifactRef | None = None,
    *,
    errors: Mapping[CandidateRef, Sequence[float]] | None = None,
) -> EvaluationCampaign:
    """A campaign whose grid ran whole, not yet closed, the contender erring less on every unit.

    The contender's artifact is ``kept`` at the cell the design retains; the control keeps none,
    as an arm that exists to be measured against. ``errors`` names each candidate's error per
    unit where a test is about the figures rather than the grid.
    """
    scored = {CONTROL: (6.0, 8.0, 10.0), CONTENDER: (3.0, 4.0, 5.0)} if errors is None else errors
    whole = campaign()
    for cell in whole.design.cells():
        retained = kept if cell.candidate == CONTENDER and whole.design.retains(cell) else None
        whole = whole.record(
            result(
                cell.candidate, cell.budget, cell.seed, scored[cell.candidate], artifact=retained
            )
        )
    return whole


def closed_campaign(kept: ArtifactRef | None = None) -> EvaluationCampaign:
    """The campaign of ``ran_campaign``, closed at ``CLOSED_AT``."""
    return ran_campaign(kept).complete(CLOSED_AT)


ROCKET = CandidateRef("minirocket")
FINER = CandidateRef("minirocket@grid_resolution=2")
SELECTED_BY = CampaignId(UUID(int=3))


def baseline(ref: CandidateRef, resolution: float = 1.0) -> CampaignCandidate:
    """A classical candidate, its grid at ``resolution``, as a catalogue would describe it."""
    return CampaignCandidate(
        ref=ref,
        kind=CandidateKind.CLASSICAL,
        budget=None,
        method=CandidateMethod.of(method="random_convolutions", grid_resolution=resolution),
        starts_from=None,
    )


def selection(errors: dict[CandidateRef, tuple[float, ...]] | None = None) -> EvaluationCampaign:
    """A finished selection between the default grid and a finer one, three repeats each.

    Without ``errors`` the finer grid errs less by far more than the spread of the repeats.
    """
    stated = errors or {ROCKET: (10.0, 10.2, 9.8), FINER: (6.0, 6.1, 5.9)}
    grid = EvaluationCampaign.designed(
        campaign_id=SELECTED_BY,
        task=TASK,
        purpose=RunPurpose.SELECTION,
        tier=ComputeTier.S,
        design=CampaignDesign(
            candidates=(baseline(ROCKET), baseline(FINER, 2.0)),
            control=ROCKET,
            endpoint=FINER,
            budgets=BUDGETS,
            endpoint_budget=LabelBudget.of(200),
            seeds=(1, 2, 3),
            rules=RULES,
            bootstrap=BOOTSTRAP,
            inner_holdout=InnerHoldout(one_in=5),
        ),
        opened_at=OPENED_AT,
    )
    for run in grid.design.cells():
        error = stated[run.candidate][run.seed - 1]
        grid = grid.record(result(run.candidate, run.budget, run.seed, (error,)))
    return grid.complete(OPENED_AT)
