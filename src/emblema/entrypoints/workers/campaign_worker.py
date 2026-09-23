from collections.abc import Mapping
from typing import Self

from emblema.config.lora_settings import LoraSettings
from emblema.config.schedule_settings import ScheduleSettings
from emblema.config.settings import Settings
from emblema.entrypoints.workers.composition_root import CompositionRoot
from emblema.evaluation.application.use_cases.run_campaign_cell import RunCampaignCellCommand
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec
from emblema.shared.jobs.job_argument import JobArgument
from emblema.shared.ports.job_queue import JobQueue


class CampaignWorker:
    """What the worker process does with a job, apart from the queue that delivered it.

    The composition root is built once and held, because the backbone it reads out of the store
    is what makes building one expensive. Everything else here is deserialisation: the settings
    and the job carry scalars, and this turns them back into the values a process and a use case
    take. No decision is made here — a cell already recorded is recognised by the use case, not
    by this.
    """

    def __init__(self, root: CompositionRoot) -> None:
        self._root = root

    @classmethod
    def from_environment(  # pragma: no cover - environment
        cls, jobs: JobQueue | None = None
    ) -> Self:
        """The worker the environment describes, submitting through ``jobs`` where one is given.

        Args:
            jobs: Where this process hands cells of its own; the configured broker unless given.

        Raises:
            ValueError: If the settings name no worker, store, database or broker, or the
                backbone is not written as the registry holds it.
        """
        settings = Settings()
        worker = settings.require_worker()
        return cls(
            CompositionRoot(
                settings,
                workspace=worker.workspace,
                corpora=worker.corpora,
                schedule=cls.schedule_of(worker.schedule),
                lora=cls.lora_of(worker.lora),
                backbone=worker.backbone_ref(),
                device=worker.device,
                jobs=jobs,
            )
        )

    def run_campaign_cell(self, arguments: Mapping[str, JobArgument]) -> None:
        """Run the cell a job names.

        Raises:
            ValueError: If the job does not name a cell.
            InvalidLabelBudgetError: If the budget it names is not one.
            CampaignNotFoundError: If the campaign is unknown.
            UnknownCampaignCellError: If the cell is not one of the campaign's grid.
        """
        self._root.services.run_campaign_cell(self.command_of(arguments))

    @staticmethod
    def command_of(arguments: Mapping[str, JobArgument]) -> RunCampaignCellCommand:
        """The coordinates a job carries, read back into a command.

        Public because reading a message back is what this entry point is for, and what a
        queue delivers is the one input nobody in this process wrote. Every coordinate is
        required, the budget included: a message that lost it would otherwise be run at a
        budget nobody asked for rather than refused.

        Raises:
            ValueError: If a coordinate is missing or is not what it has to be.
            InvalidLabelBudgetError: If the budget is neither a count nor the word for every
                window there is.
        """
        campaign, candidate = arguments.get("campaign"), arguments.get("candidate")
        budget, seed = arguments.get("budget"), arguments.get("seed")
        if not isinstance(campaign, str) or not isinstance(candidate, str):
            raise ValueError("a cell names its campaign and its candidate")
        if not isinstance(budget, str):
            raise ValueError("a cell names the budget of labels it learns from")
        if not isinstance(seed, int):
            raise ValueError("a cell names the seed of its repeat")
        return RunCampaignCellCommand(
            campaign=CampaignId.parse(campaign),
            cell=CampaignCell(
                candidate=CandidateRef(candidate),
                budget=LabelBudget.parse(budget),
                seed=seed,
            ),
        )

    @staticmethod
    def schedule_of(settings: ScheduleSettings) -> AdaptationSchedule:
        """How every cell of this process's campaigns learns, as the environment declares it.

        Raises:
            InvalidAdaptationScheduleError: If what it declares is not a schedule that stands up.
        """
        return AdaptationSchedule(
            epochs=settings.epochs,
            min_steps=settings.min_steps,
            batch_size=settings.batch_size,
            learning_rate=settings.learning_rate,
            weight_decay=settings.weight_decay,
            warmup_fraction=settings.warmup_fraction,
            final_lr_fraction=settings.final_lr_fraction,
        )

    @staticmethod
    def lora_of(settings: LoraSettings) -> LoraSpec:
        """The low-rank updates the arm of that name adds, as the environment declares them.

        Raises:
            InvalidLoraSpecError: If what it declares is not a specification that stands up.
        """
        return LoraSpec(
            rank=settings.rank,
            alpha=settings.alpha,
            dropout=settings.dropout,
            targets=tuple(
                target.strip() for target in settings.targets.split(",") if target.strip()
            ),
        )
