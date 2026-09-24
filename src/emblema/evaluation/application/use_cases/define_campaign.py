from collections.abc import Mapping
from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef, TaskId
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.campaign_design import CampaignDesign
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.exceptions import TunedChoiceMismatchError
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.statistics.comparison_rules import ComparisonRules
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.domain.tuning.tuned_choice import TunedChoice
from emblema.evaluation.ports.candidate_catalogue import CandidateCatalogue
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository
from emblema.shared.kernel.compute import ComputeTier
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.id_generator import IdGenerator


@dataclass(frozen=True, kw_only=True)
class DefineCampaignCommand:
    """A comparison stated in full before any of it runs.

    Attributes:
        task: Task every candidate answers.
        purpose: What the campaign is for; only the final one may see the task's frozen side.
        tier: Hardware class every run is measured on, declared rather than inferred from
            whatever machine happens to pick the work up.
        candidates: Who competes, in reporting order, the control among them.
        control: Which of them the others are measured against.
        endpoint: Which candidate the campaign's single claim is about.
        budgets: How many labelled windows each cell learns from.
        endpoint_budget: The budget the claim is made at.
        seeds: Repeats of every cell; the first is the one whose fitted candidate is kept.
        rules: What a verdict requires, as the campaign's registration stated it.
        bootstrap: How the interval around each comparison is drawn.
        inner_holdout: How the tuning side is divided, for a campaign that selects among
            variants; ``None`` for one that compares.
        tuned: Which variant a candidate runs at a budget, each naming the selection that
            chose it.
    """

    task: TaskId
    purpose: RunPurpose
    tier: ComputeTier
    candidates: tuple[CandidateRef, ...]
    control: CandidateRef
    endpoint: CandidateRef
    budgets: tuple[LabelBudget, ...]
    endpoint_budget: LabelBudget
    seeds: tuple[int, ...]
    rules: ComparisonRules
    bootstrap: PairedUnitBootstrap
    inner_holdout: InnerHoldout | None = None
    tuned: tuple[TunedChoice, ...] = ()


class DefineCampaign:
    """Lays out a campaign's grid and stores it, with nothing run.

    What each candidate is comes from the catalogue rather than from the caller: the compute
    budget a candidate is held to has to be the one it will actually spend, and a figure written
    down beside a candidate can disagree with what the candidate does. Asking here means the
    disagreement is impossible rather than merely unlikely. A catalogue and not a provider,
    because declaring a comparison never runs one.

    A tuned choice is checked, not trusted. The selection it names is read again, by its own
    rule, and a design naming any variant but the one the selection chose is refused — so the
    variant a comparison runs is the output of a procedure declared before it ran, and cannot
    be one picked afterwards by whoever wrote the file.
    """

    def __init__(
        self,
        tasks: DownstreamTaskRepository,
        campaigns: EvaluationCampaignRepository,
        candidates: CandidateCatalogue,
        ids: IdGenerator,
        clock: Clock,
    ) -> None:
        self._tasks = tasks
        self._campaigns = campaigns
        self._candidates = candidates
        self._ids = ids
        self._clock = clock

    def __call__(self, command: DefineCampaignCommand) -> CampaignId:
        """Describe every candidate, lay out the grid and store it.

        Raises:
            TaskNotFoundError: If the task is unknown.
            ProtocolMismatchError: If the task spends no labels, so it has no budget axis to
                spread a grid over.
            UnknownCandidateError: If the provider supplies none of that name.
            InvalidCampaignDesignError: If the design contradicts itself.
            CampaignNotFoundError: If a tuned choice names a selection that is not stored.
            SelectionNotReadableError: If that selection has not finished or holds no choice
                for the pairing.
            TunedChoiceMismatchError: If it chose another variant, or ran over another task.
        """
        task = self._tasks.get(command.task)
        task.accept_campaign()
        described = {
            ref: self._candidates.describe(ref)
            for ref in dict.fromkeys(
                (*command.candidates, *(choice.variant for choice in command.tuned))
            )
        }
        for choice in command.tuned:
            self._check(choice, task.task_id, described)
        campaign = EvaluationCampaign.designed(
            campaign_id=self._ids.generate(CampaignId),
            task=task.task_id,
            purpose=command.purpose,
            tier=command.tier,
            design=CampaignDesign(
                candidates=tuple(described[ref] for ref in command.candidates),
                control=command.control,
                endpoint=command.endpoint,
                budgets=command.budgets,
                endpoint_budget=command.endpoint_budget,
                seeds=command.seeds,
                rules=command.rules,
                bootstrap=command.bootstrap,
                inner_holdout=command.inner_holdout,
                tuned=command.tuned,
                variants=tuple(
                    described[ref]
                    for ref in dict.fromkeys(choice.variant for choice in command.tuned)
                ),
            ),
            opened_at=self._clock.now(),
        )
        self._campaigns.save(campaign, seen=campaign.revision)
        return campaign.campaign_id

    def _check(
        self,
        choice: TunedChoice,
        task: TaskId,
        described: Mapping[CandidateRef, CampaignCandidate],
    ) -> None:
        """Refuse a choice its selection, read again by its rule, did not make.

        A name is not enough: the variant and the candidate it varies must be described here
        exactly as the selection described them when it ran, or a change of configuration in
        between would run another model under the name the selection chose.

        Raises:
            CampaignNotFoundError: If the selection is not stored.
            SelectionNotReadableError: If it cannot choose for that pairing.
            TunedChoiceMismatchError: If it chose otherwise, selected over another task, or
                described the variant or its base otherwise than this process does.
        """
        selection = self._campaigns.get(choice.selected_by)
        if selection.task != task:
            raise TunedChoiceMismatchError(
                f"selection {choice.selected_by} ran over task {selection.task}, not {task}"
            )
        chosen = selection.selected(choice.candidate, choice.budget)
        if chosen != choice.variant:
            raise TunedChoiceMismatchError(
                f"selection {choice.selected_by} chose {chosen} for {choice.candidate} at "
                f"{choice.budget}, not {choice.variant}"
            )
        for ref in (choice.candidate, choice.variant):
            ran = selection.design.get_candidate(ref)
            if ran != described[ref]:
                raise TunedChoiceMismatchError(
                    f"selection {choice.selected_by} ran {ref} as {ran}, and this process "
                    f"describes it as {described[ref]}"
                )
