"""A comparison that runs tuned variants is declared only if its selection chose them."""

from dataclasses import replace
from uuid import UUID

import pytest

from emblema.evaluation.adapters.candidates.classical_arm import ClassicalArm
from emblema.evaluation.adapters.candidates.classical_baseline_catalogue import (
    ClassicalBaselineCatalogue,
)
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.application.use_cases.define_campaign import (
    DefineCampaign,
    DefineCampaignCommand,
)
from emblema.evaluation.application.use_cases.select_tuned_variants import (
    SelectTunedVariants,
    SelectTunedVariantsCommand,
)
from emblema.evaluation.contracts.identifiers import CandidateRef, TaskId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.campaign_reading import CampaignReading
from emblema.evaluation.domain.classical.boosted_trees import BoostedTrees
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.evaluation.domain.exceptions import (
    SelectionNotReadableError,
    TunedChoiceMismatchError,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.domain.tuning.tuned_choice import TunedChoice
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.kernel.compute import ComputeTier
from tests.evaluation.support import (
    BOOTSTRAP,
    BUDGETS,
    FINER,
    OPENED_AT,
    ROCKET,
    RULES,
    SELECTED_BY,
    boosting,
    convolutions,
    reopened,
    selection,
    store,
    task,
)

OTHER = CandidateRef("other")
AT_50, AT_200 = BUDGETS


class Process:
    def __init__(self, stored: CampaignReading | None = None) -> None:
        """The context over in-memory adapters, ``stored`` being the selection it holds."""
        self.tasks = InMemoryDownstreamTaskRepository()
        self.tasks.save(task())
        self.campaigns = InMemoryEvaluationCampaignRepository()
        catalogue = ClassicalBaselineCatalogue(
            (
                ClassicalArm(ref=ROCKET, method=convolutions(), sources=()),
                ClassicalArm(ref=OTHER, method=convolutions(), sources=()),
            )
        )
        store(self.campaigns, described_by(catalogue) if stored is None else stored)
        self.define = DefineCampaign(
            self.tasks, self.campaigns, catalogue, SequentialIdGenerator(), FixedClock(OPENED_AT)
        )
        self.select = SelectTunedVariants(self.campaigns)

    def declared(self, *tuned: TunedChoice) -> DefineCampaignCommand:
        return DefineCampaignCommand(
            task=task().task_id,
            purpose=RunPurpose.TUNING,
            tier=ComputeTier.S,
            candidates=(OTHER, ROCKET),
            control=OTHER,
            endpoint=ROCKET,
            budgets=BUDGETS,
            endpoint_budget=AT_200,
            seeds=(1, 2),
            rules=RULES,
            bootstrap=BOOTSTRAP,
            tuned=tuned,
        )


def described_by(catalogue: ClassicalBaselineCatalogue) -> CampaignReading:
    """The finished selection, its candidates described as ``catalogue`` describes them."""
    chosen = selection()
    return replace(
        chosen,
        campaign=replace(
            chosen.campaign,
            design=replace(
                chosen.campaign.design,
                candidates=(catalogue.describe(ROCKET), catalogue.describe(FINER)),
            ),
        ),
    )


def chose(variant: CandidateRef, budget: LabelBudget = AT_200) -> TunedChoice:
    return TunedChoice(candidate=ROCKET, budget=budget, variant=variant, selected_by=SELECTED_BY)


def test_what_a_selection_chose_is_read_per_budget_as_the_comparison_names_it() -> None:
    chosen = Process().select(SelectTunedVariantsCommand(campaign=SELECTED_BY, candidate=ROCKET))

    assert chosen == (chose(FINER, AT_50), chose(FINER, AT_200))


def test_a_comparison_naming_what_the_selection_chose_runs_that_variant_at_that_budget() -> None:
    process = Process()

    declared = process.define(process.declared(chose(FINER)))

    grid = process.campaigns.get(declared)
    tuned = grid.evaluation_of(CampaignCell(candidate=ROCKET, budget=AT_200, seed=1))
    untuned = grid.evaluation_of(CampaignCell(candidate=ROCKET, budget=AT_50, seed=1))
    assert tuned.declared.ref == FINER
    stated = {p.name: p.value for p in tuned.declared.method.parameters}
    assert stated["grid_resolution"] == "2.0"
    assert untuned.declared.ref == ROCKET


def test_a_comparison_naming_a_variant_its_selection_did_not_choose_is_refused() -> None:
    process = Process()

    with pytest.raises(TunedChoiceMismatchError, match="chose"):
        process.define(process.declared(chose(CandidateRef("minirocket@grid_resolution=3"))))


def test_a_comparison_naming_an_unfinished_selection_is_refused() -> None:
    process = Process(reopened(selection()))

    with pytest.raises(SelectionNotReadableError):
        process.define(process.declared(chose(FINER)))


def test_a_selection_over_another_task_cannot_tune_this_comparison() -> None:
    process = Process()
    elsewhere = replace(task(), task_id=TaskId(UUID(int=99)))
    process.tasks.save(elsewhere)

    with pytest.raises(TunedChoiceMismatchError, match="ran over task"):
        process.define(replace(process.declared(chose(FINER)), task=elsewhere.task_id))


TREES = CandidateRef("boosted_trees_per_channel")
SHALLOW = CandidateRef("boosted_trees_per_channel@max_depth=2")
DEEP = CandidateRef("boosted_trees_per_channel@max_depth=6")
SLOW = CandidateRef("boosted_trees_per_channel@learning_rate=0.1,max_depth=2")


def trees_catalogue(**boosting_overrides: float) -> ClassicalBaselineCatalogue:
    return ClassicalBaselineCatalogue(
        (
            ClassicalArm(
                ref=TREES,
                method=BoostedTrees(
                    features=FeatureScheme.PER_CHANNEL, boosting=boosting(**boosting_overrides)
                ),
                sources=(),
            ),
            ClassicalArm(ref=OTHER, method=convolutions(), sources=()),
        )
    )


def around_shallow_trees(catalogue: ClassicalBaselineCatalogue) -> CampaignReading:
    """A finished selection declared around the shallow trees, with no bare name in its grid."""
    return selection(
        errors={SHALLOW: (10.0, 10.2, 9.8), DEEP: (10.1, 10.3, 9.9), SLOW: (10.2, 10.4, 10.0)},
        candidates=tuple(catalogue.describe(ref) for ref in (SHALLOW, DEEP, SLOW)),
    )


def test_a_comparison_tuned_by_a_selection_declared_around_a_variant_is_defined() -> None:
    catalogue = trees_catalogue()
    process = Process(around_shallow_trees(catalogue))
    define = DefineCampaign(
        process.tasks, process.campaigns, catalogue, SequentialIdGenerator(), FixedClock(OPENED_AT)
    )
    command = replace(
        process.declared(
            TunedChoice(candidate=TREES, budget=AT_200, variant=SHALLOW, selected_by=SELECTED_BY)
        ),
        candidates=(OTHER, TREES),
        endpoint=TREES,
    )

    declared = define(command)

    grid = process.campaigns.get(declared)
    assert (
        grid.evaluation_of(CampaignCell(candidate=TREES, budget=AT_200, seed=1)).declared.ref
        == SHALLOW
    )
    assert (
        grid.evaluation_of(CampaignCell(candidate=TREES, budget=AT_50, seed=1)).declared.ref
        == TREES
    )


def test_a_setting_the_selection_turned_around_under_another_configuration_is_refused() -> None:
    process = Process(around_shallow_trees(trees_catalogue()))
    # The same names, described by a process whose trees take more rounds.
    define = DefineCampaign(
        process.tasks,
        process.campaigns,
        trees_catalogue(rounds=16),
        SequentialIdGenerator(),
        FixedClock(OPENED_AT),
    )
    command = replace(
        process.declared(
            TunedChoice(candidate=TREES, budget=AT_200, variant=SHALLOW, selected_by=SELECTED_BY)
        ),
        candidates=(OTHER, TREES),
        endpoint=TREES,
    )

    with pytest.raises(TunedChoiceMismatchError, match="describes it as"):
        define(command)


def test_a_variant_the_selection_ran_under_another_configuration_is_refused() -> None:
    # The same names, another model: the selection ran the grids over fewer features than this
    # process would, so what it chose is not what the comparison would run.
    elsewhere = ClassicalBaselineCatalogue(
        (
            ClassicalArm(
                ref=ROCKET, method=convolutions().tuned("convolution_features", "168"), sources=()
            ),
        )
    )
    process = Process(described_by(elsewhere))

    with pytest.raises(TunedChoiceMismatchError, match="describes it as"):
        process.define(process.declared(chose(FINER)))
