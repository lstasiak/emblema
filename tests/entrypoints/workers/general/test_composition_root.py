"""The worker that competes the baselines, assembled over adapters that touch nothing.

What a composition test can hold is the wiring: that the use cases share one registry, that a
process given no overrides reaches the bucket, the database and the broker its settings name,
and that one left without the knobs it fits by fails as it is assembled rather than when it is
first used. The one thing peculiar to this process is what it must not have: the machine-learning
stack, which is why its image is the smaller and why a campaign of baselines needs none of it.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from emblema.config.boosting_settings import BoostingSettings
from emblema.entrypoints.workers.general.composition_root import CompositionRoot
from emblema.entrypoints.workers.known_baselines import KnownBaselines
from emblema.evaluation.adapters.in_memory.candidate_provider import InMemoryCandidateProvider
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.adapters.persistence.downstream_task_repository import (
    SqlAlchemyDownstreamTaskRepository,
)
from emblema.evaluation.adapters.persistence.evaluation_campaign_repository import (
    SqlAlchemyEvaluationCampaignRepository,
)
from emblema.evaluation.application.use_cases.advance_campaign import AdvanceCampaignCommand
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.queues.celery_job_queue import CeleryJobQueue
from emblema.shared.adapters.queues.immediate_job_queue import ImmediateJobQueue
from emblema.shared.adapters.storage.s3 import S3ArtifactStore
from emblema.shared.ports.exceptions import JobQueueError
from tests.evaluation.support import boosting, campaign, convolutions, task
from tests.support.settings import unreachable_store

DECLARED = BoostingSettings(
    rounds=8,
    max_depth=3,
    learning_rate=0.3,
    row_share=1.0,
    feature_share=1.0,
    min_leaf_weight=1.0,
    l2_penalty=1.0,
    threads=1,
)


def process(tmp_path: Path) -> tuple[CompositionRoot, InMemoryEvaluationCampaignRepository]:
    tasks = InMemoryDownstreamTaskRepository()
    tasks.save(task())
    campaigns = InMemoryEvaluationCampaignRepository()
    campaigns.save(campaign(), seen=0)
    root = CompositionRoot.over(
        store=InMemoryArtifactStore(),
        tasks=tasks,
        campaigns=campaigns,
        candidates=InMemoryCandidateProvider((), (), lambda _: ()),
        jobs=ImmediateJobQueue({}),
        workspace=tmp_path,
        corpora=tmp_path,
    )
    return root, campaigns


def test_the_use_cases_share_the_registries_the_process_was_given(tmp_path: Path) -> None:
    root, campaigns = process(tmp_path)

    assert root.adapters.campaigns is campaigns
    with pytest.raises(JobQueueError, match="runs no job"):
        root.services.advance_campaign(AdvanceCampaignCommand(campaign=campaign().campaign_id))


def test_without_overrides_the_process_runs_on_what_the_settings_name(tmp_path: Path) -> None:
    root = CompositionRoot(
        unreachable_store(),
        workspace=tmp_path,
        corpora=tmp_path,
        candidates=InMemoryCandidateProvider((), (), lambda _: ()),
    )

    assert isinstance(root.adapters.store, S3ArtifactStore)
    assert isinstance(root.adapters.tasks, SqlAlchemyDownstreamTaskRepository)
    assert isinstance(root.adapters.campaigns, SqlAlchemyEvaluationCampaignRepository)
    assert isinstance(root.adapters.jobs, CeleryJobQueue)


def test_a_process_bringing_neither_settings_nor_a_store_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="store"):
        CompositionRoot(workspace=tmp_path, corpora=tmp_path)


def test_a_process_left_to_build_baselines_without_the_knobs_to_fit_them_is_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="classical knobs"):
        CompositionRoot(
            unreachable_store(),
            workspace=tmp_path,
            corpora=tmp_path,
            store=InMemoryArtifactStore(),
            tasks=InMemoryDownstreamTaskRepository(),
            campaigns=InMemoryEvaluationCampaignRepository(),
            jobs=ImmediateJobQueue({}),
        )


def test_left_to_build_its_own_candidates_the_process_serves_every_baseline(
    tmp_path: Path,
) -> None:
    root = CompositionRoot(
        unreachable_store(),
        workspace=tmp_path,
        corpora=tmp_path,
        boosting=boosting(threads=2),
        convolutions=convolutions(),
        store=InMemoryArtifactStore(),
        tasks=InMemoryDownstreamTaskRepository(),
        campaigns=InMemoryEvaluationCampaignRepository(),
        jobs=ImmediateJobQueue({}),
    )

    described = root.adapters.candidates.describe(KnownBaselines.ACROSS_CHANNELS)

    assert described.starts_from is None
    assert described.budget is None
    stated = {parameter.name: parameter.value for parameter in described.method.parameters}
    assert stated["features"] == "channel_aggregated"
    assert stated["threads"] == "2"
    rocket = root.adapters.candidates.describe(KnownBaselines.MINIROCKET).method.parameters
    assert {parameter.name: parameter.value for parameter in rocket}["method"] == (
        "random_convolutions"
    )


def test_assembling_this_process_imports_no_machine_learning_stack() -> None:
    # The claim the whole split rests on, and the only way to hold it is to look at a process
    # that did it: an import inside this one would be satisfied by whatever a test imported
    # first. A campaign of baselines has to run where torch is not installed at all.
    read = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import emblema.entrypoints.workers.general.composition_root  # noqa: F401\n"
            "print(sorted(m for m in sys.modules if m in {'torch', 'onnxruntime'}))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert read.stdout.strip() == "[]"
