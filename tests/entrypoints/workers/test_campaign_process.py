from pathlib import Path

from emblema.entrypoints.workers.campaign_process import CampaignProcess
from emblema.evaluation.adapters.in_memory.candidate_provider import InMemoryCandidateProvider
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.application.use_cases.complete_campaign import CompleteCampaignCommand
from emblema.serving.adapters.in_memory.promotable_artifact_repository import (
    InMemoryPromotableArtifactRepository,
)
from emblema.serving.adapters.persistence.promotable_artifact_repository import (
    SqlAlchemyPromotableArtifactRepository,
)
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.queues.immediate_job_queue import ImmediateJobQueue
from tests.evaluation.support import CAMPAIGN, CONTENDER, artifact, ran_campaign
from tests.support.settings import unreachable_store


def test_a_campaign_closed_in_this_process_is_heard_by_serving(tmp_path: Path) -> None:
    campaigns = InMemoryEvaluationCampaignRepository()
    kept = artifact("contender")
    campaigns.save(ran_campaign(kept), seen=0)
    process = CampaignProcess(
        workspace=tmp_path,
        corpora=tmp_path,
        store=InMemoryArtifactStore(),
        tasks=InMemoryDownstreamTaskRepository(),
        campaigns=campaigns,
        promotables=InMemoryPromotableArtifactRepository(),
        jobs=ImmediateJobQueue({}),
    )
    adapters, services = process.assemble(InMemoryCandidateProvider((), (), lambda _: ()))

    services.complete_campaign(CompleteCampaignCommand(campaign=CAMPAIGN))

    (promotable,) = adapters.promotables.find_by_checksum(kept.checksum)
    assert promotable.origin.candidate == CONTENDER


def test_left_to_build_it_the_process_keeps_the_projection_in_the_database(
    tmp_path: Path,
) -> None:
    process = CampaignProcess(
        unreachable_store(),
        workspace=tmp_path,
        corpora=tmp_path,
        store=InMemoryArtifactStore(),
        jobs=ImmediateJobQueue({}),
    )

    assert isinstance(process.promotables, SqlAlchemyPromotableArtifactRepository)
