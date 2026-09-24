from pathlib import Path
from typing import Self

from emblema.config.settings import Settings
from emblema.entrypoints.cli.campaign.adapters import Adapters
from emblema.entrypoints.cli.campaign.services import Services
from emblema.entrypoints.workers.campaign_process import CampaignProcess
from emblema.entrypoints.workers.declared_worker import DeclaredWorker
from emblema.entrypoints.workers.known_arms import KnownArms
from emblema.entrypoints.workers.known_baselines import KnownBaselines
from emblema.evaluation.adapters.candidates.routed_candidate_catalogue import (
    RoutedCandidateCatalogue,
)
from emblema.evaluation.application.assemblers.campaign_completed_assembler import (
    CampaignCompletedAssembler,
)
from emblema.evaluation.application.use_cases.advance_campaign import AdvanceCampaign
from emblema.evaluation.application.use_cases.announce_campaign import AnnounceCampaign
from emblema.evaluation.application.use_cases.complete_campaign import CompleteCampaign
from emblema.evaluation.application.use_cases.define_campaign import DefineCampaign
from emblema.evaluation.application.use_cases.define_downstream_task import DefineDownstreamTask
from emblema.evaluation.domain.classical.gradient_boosting_spec import GradientBoostingSpec
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec
from emblema.evaluation.ports.candidate_catalogue import CandidateCatalogue
from emblema.shared.kernel.artifacts import ArtifactRef


class CompositionRoot:
    """Assembles the process that declares a campaign and hands its cells to the queues.

    It competes nobody itself and runs no cell. What it needs the candidates for is the one
    thing a design cannot be written without: what each of them is. A budget written down beside
    a candidate can disagree with what the candidate does, so the design asks whoever supplies
    it — which here means the catalogues of both kinds, behind the same routing a worker's
    campaign will be read through.

    Catalogues and not providers, so this process carries neither the training stack nor the one
    the baselines are fitted with — which on the platform this is developed on is the condition
    under which it can exist, since the two cannot share a process.

    Attributes:
        adapters: The port implementations the process runs on.
        services: The use cases it can run.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        workspace: Path,
        corpora: Path,
        schedule: AdaptationSchedule,
        lora: LoraSpec,
        backbone: ArtifactRef,
        boosting: GradientBoostingSpec,
    ) -> None:
        """Assemble the process from what the campaign's candidates are set to.

        Args:
            settings: Values read from the environment; only the root and the adapters see them.
            workspace: Directory corpus blocks are fetched to and mapped from.
            corpora: Directory the raw corpora sit in, where a task's labels are read from.
            schedule: What the neural arms learn under, which the design records.
            lora: The low-rank updates the arm of that name adds.
            backbone: Weights the pretrained arms start from.
            boosting: How hard the baselines fit, which the design records too.

        Raises:
            ValueError: If the settings name no store, database or broker.
        """
        process = CampaignProcess(settings, workspace=workspace, corpora=corpora)
        catalogue = self._candidates(schedule, lora, backbone, boosting)
        outcomes = CampaignCompletedAssembler()
        self.adapters = Adapters(
            corpus=process.corpus,
            candidates=catalogue,
            tasks=process.tasks,
            campaigns=process.campaigns,
            jobs=process.jobs,
        )
        self.services = Services(
            define_downstream_task=DefineDownstreamTask(process.tasks, process.corpus, process.ids),
            define_campaign=DefineCampaign(
                process.tasks,
                process.campaigns,
                catalogue,
                process.ids,
                process.clock,
            ),
            advance_campaign=AdvanceCampaign(
                process.campaigns,
                process.jobs,
                CompleteCampaign(
                    process.campaigns, outcomes, process.clock, process.ids, process.events
                ),
            ),
            announce_campaign=AnnounceCampaign(
                process.campaigns, outcomes, process.ids, process.events
            ),
        )

    @classmethod
    def from_environment(cls, settings: Settings | None = None) -> Self:  # pragma: no cover - env
        """The process the environment describes.

        Raises:
            ValueError: If the settings name no worker, store, database or broker, or leave out
                anything the campaign's candidates are declared by.
        """
        read = Settings() if settings is None else settings
        worker = read.require_worker()
        declared = DeclaredWorker(worker)
        return cls(
            read,
            workspace=worker.workspace,
            corpora=worker.corpora,
            schedule=declared.schedule(),
            lora=declared.lora(),
            backbone=worker.require_backbone_ref(),
            boosting=declared.boosting(),
        )

    @staticmethod
    def _candidates(
        schedule: AdaptationSchedule,
        lora: LoraSpec,
        backbone: ArtifactRef,
        boosting: GradientBoostingSpec,
    ) -> CandidateCatalogue:
        """Every candidate a campaign may name, each routed to whoever holds it."""
        arms = KnownArms.catalogue(backbone, lora, schedule)
        baselines = KnownBaselines.catalogue(boosting)
        return RoutedCandidateCatalogue(
            {
                **dict.fromkeys(KnownArms.refs(), arms),
                **dict.fromkeys(KnownBaselines.refs(), baselines),
            }
        )
