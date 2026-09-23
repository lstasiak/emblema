from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef, TaskId
from emblema.evaluation.domain.campaign.campaign_design import CampaignDesign
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.statistics.comparison_rules import ComparisonRules
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.ports.candidate_provider import CandidateProvider
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


class DefineCampaign:
    """Lays out a campaign's grid and stores it, with nothing run.

    What each candidate is comes from the provider rather than from the caller: the compute
    budget a candidate is held to has to be the one it will actually spend, and a figure written
    down beside a candidate can disagree with what the candidate does. Asking here means the
    disagreement is impossible rather than merely unlikely.
    """

    def __init__(
        self,
        tasks: DownstreamTaskRepository,
        campaigns: EvaluationCampaignRepository,
        candidates: CandidateProvider,
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
        """
        task = self._tasks.get(command.task)
        task.accept_campaign()
        campaign = EvaluationCampaign.designed(
            campaign_id=self._ids.generate(CampaignId),
            task=task.task_id,
            purpose=command.purpose,
            tier=command.tier,
            design=CampaignDesign(
                candidates=tuple(self._candidates.describe(ref) for ref in command.candidates),
                control=command.control,
                endpoint=command.endpoint,
                budgets=command.budgets,
                endpoint_budget=command.endpoint_budget,
                seeds=command.seeds,
                rules=command.rules,
                bootstrap=command.bootstrap,
            ),
            opened_at=self._clock.now(),
        )
        self._campaigns.save(campaign, seen=campaign.revision)
        return campaign.campaign_id
