"""The worker that adapts a backbone, assembled over adapters that touch nothing.

What a composition test can hold is the wiring: that the use cases share one registry, that a
process given no overrides reaches the bucket, the database and the broker its settings name,
and that one left without either fails as it is assembled rather than when it is first used.
The work itself is exercised by the campaign cycle, which needs no process.
"""

from pathlib import Path
from typing import Any

import pytest

from emblema.config.lora_settings import LoraSettings
from emblema.config.schedule_settings import ScheduleSettings
from emblema.entrypoints.workers.campaign_worker import CampaignWorker
from emblema.entrypoints.workers.known_arms import KnownArms
from emblema.entrypoints.workers.known_patch_models import KnownPatchModels
from emblema.entrypoints.workers.ml.composition_root import CompositionRoot
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
from emblema.evaluation.application.use_cases.open_test_split import OpenTestSplitCommand
from emblema.evaluation.contracts.events import FrozenTestSplitOpened
from emblema.evaluation.domain.exceptions import (
    FrozenTestSplitClosedError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.queues.celery_job_queue import CeleryJobQueue
from emblema.shared.adapters.queues.immediate_job_queue import ImmediateJobQueue
from emblema.shared.adapters.storage.s3 import S3ArtifactStore
from emblema.shared.ports.exceptions import JobQueueError
from tests.entrypoints.test_restored_backbones import stored
from tests.evaluation.support import (
    CONTENDER,
    CONTROL,
    LORA,
    WEIGHTS,
    adaptation_schedule,
    campaign,
    candidate,
    patch_spec,
    task,
)
from tests.support.settings import unreachable_store

CAMPAIGN_TEXT = "00000000-0000-0000-0000-000000000002"
SCHEDULE = adaptation_schedule()
DECLARED_SCHEDULE = ScheduleSettings(
    epochs=2,
    min_steps=0,
    batch_size=2,
    learning_rate=1e-2,
    weight_decay=0.0,
    warmup_fraction=0.0,
    final_lr_fraction=1.0,
)
DECLARED_LORA = LoraSettings(
    rank=2, alpha=4.0, dropout=0.0, targets="qkv, attention.projection, feedforward"
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
        candidates=InMemoryCandidateProvider(
            (candidate(CONTROL), candidate(CONTENDER)),
            (UnitKey("c"),),
            lambda _: (2.0,),
        ),
        jobs=ImmediateJobQueue({}),
        workspace=tmp_path,
        corpora=tmp_path,
        schedule=SCHEDULE,
    )
    return root, campaigns


def test_the_use_cases_share_the_registries_the_process_was_given(tmp_path: Path) -> None:
    root, campaigns = process(tmp_path)

    assert root.adapters.campaigns is campaigns
    # Reaching the scheduler at all is the proof: the campaign this use case expanded into cells
    # is the one saved into the registry the process was handed.
    with pytest.raises(JobQueueError, match="runs no job"):
        root.services.advance_campaign(AdvanceCampaignCommand(campaign=campaign().campaign_id))


def test_without_overrides_the_process_runs_on_what_the_settings_name(tmp_path: Path) -> None:
    root = CompositionRoot(
        unreachable_store(),
        workspace=tmp_path,
        corpora=tmp_path,
        backbone=WEIGHTS,
        schedule=SCHEDULE,
        candidates=InMemoryCandidateProvider((), (), lambda _: ()),
    )

    assert isinstance(root.adapters.store, S3ArtifactStore)
    assert isinstance(root.adapters.tasks, SqlAlchemyDownstreamTaskRepository)
    assert isinstance(root.adapters.campaigns, SqlAlchemyEvaluationCampaignRepository)
    assert isinstance(root.adapters.jobs, CeleryJobQueue)


def test_a_process_bringing_neither_settings_nor_a_store_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="store"):
        CompositionRoot(workspace=tmp_path, corpora=tmp_path, schedule=SCHEDULE)


@pytest.mark.parametrize("left_out", ["backbone", "lora", "patch"])
def test_a_process_left_to_build_candidates_without_what_they_are_built_over_is_refused(
    tmp_path: Path, left_out: str
) -> None:
    # Any: three arguments of three types, and whichever two remain are spread into the call.
    given: dict[str, Any] = {"backbone": WEIGHTS, "lora": LORA, "patch": patch_spec()}
    del given[left_out]
    with pytest.raises(ValueError, match="needs its candidates given"):
        CompositionRoot(
            unreachable_store(),
            workspace=tmp_path,
            corpora=tmp_path,
            schedule=SCHEDULE,
            **given,
            store=InMemoryArtifactStore(),
            tasks=InMemoryDownstreamTaskRepository(),
            campaigns=InMemoryEvaluationCampaignRepository(),
            jobs=ImmediateJobQueue({}),
        )


def test_the_arms_are_the_four_modes_over_the_backbone_the_process_serves() -> None:
    arms = KnownArms.over(WEIGHTS, LORA)

    assert [str(arm.ref) for arm in arms] == [str(ref) for ref in KnownArms.refs()]
    assert arms[0].backbone is None
    assert all(arm.backbone == WEIGHTS for arm in arms[1:])
    assert arms[2].lora == LORA


def test_the_frozen_side_is_opened_through_the_publisher_the_process_holds(
    tmp_path: Path,
) -> None:
    # The one operation whose whole point is the record it leaves. A process that built it out
    # of sight could give it a publisher nobody listens to, and nothing would announce that.
    heard: list[FrozenTestSplitOpened] = []
    root, _ = process(tmp_path)
    root.adapters.subscriptions.subscribe(FrozenTestSplitOpened, heard.append)

    root.services.open_test_split(
        OpenTestSplitCommand(task=task().task_id, purpose=RunPurpose.FINAL)
    )

    assert [event.task for event in heard] == [task().task_id]


def test_a_tuning_run_is_refused_the_frozen_side_and_records_nothing(tmp_path: Path) -> None:
    heard: list[FrozenTestSplitOpened] = []
    root, _ = process(tmp_path)
    root.adapters.subscriptions.subscribe(FrozenTestSplitOpened, heard.append)

    with pytest.raises(FrozenTestSplitClosedError):
        root.services.open_test_split(
            OpenTestSplitCommand(task=task().task_id, purpose=RunPurpose.TUNING)
        )

    assert heard == []


def test_a_job_the_worker_is_handed_runs_the_cell_it_names(tmp_path: Path) -> None:
    # The hop the Celery task makes: arguments in, a recorded cell out, through the process's
    # own use case rather than through anything the test stands in for.
    root, campaigns = process(tmp_path)
    stated = campaign()

    CampaignWorker(root.services.run_campaign_cell).run_campaign_cell(
        {
            "campaign": str(stated.campaign_id),
            "candidate": str(CONTROL),
            "budget": "200",
            "seed": 2,
        }
    )

    recorded = campaigns.get(stated.campaign_id).results
    assert [(r.cell.candidate, r.cell.budget, r.cell.seed) for r in recorded] == [
        (CONTROL, LabelBudget.of(200), 2)
    ]


def test_left_to_build_its_own_candidates_the_process_serves_the_backbone_it_was_given(
    tmp_path: Path,
) -> None:
    # The one path the overrides usually skip: the provider built from a store, a backbone and a
    # schedule, which is what a worker started from the environment actually runs on.
    store = InMemoryArtifactStore()
    weights, _ = stored(store)
    root = CompositionRoot(
        unreachable_store(),
        workspace=tmp_path,
        corpora=tmp_path,
        backbone=weights,
        lora=LORA,
        patch=patch_spec(),
        schedule=SCHEDULE,
        store=store,
        tasks=InMemoryDownstreamTaskRepository(),
        campaigns=InMemoryEvaluationCampaignRepository(),
        jobs=ImmediateJobQueue({}),
    )

    described = root.adapters.candidates.describe(KnownArms.LORA)

    assert described.starts_from == weights
    stated = {parameter.name: parameter.value for parameter in described.method.parameters}
    assert stated["transfer_mode"] == "lora"
    assert stated["lora_rank"] == str(LORA.rank)
    patched = root.adapters.candidates.describe(KnownPatchModels.PATCH_TRANSFORMER)
    assert patched.starts_from is None
    # One schedule for every network is what holds them to one compute budget.
    assert patched.budget == described.budget
