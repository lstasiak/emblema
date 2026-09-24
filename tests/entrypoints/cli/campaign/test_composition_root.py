"""The process that declares a campaign, assembled from what the settings name.

Its one peculiarity is what it must not have. Declaring a comparison asks each competitor what
it is and never runs one, so this process carries neither the training stack nor the one the
baselines are fitted with — which on this platform is not a saving but the condition under
which it can exist, since the two cannot share a process at all.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from emblema.entrypoints.cli.campaign.composition_root import CompositionRoot
from emblema.entrypoints.workers.known_arms import KnownArms
from emblema.entrypoints.workers.known_baselines import KnownBaselines
from emblema.evaluation.adapters.blocks.block_corpus_windows import BlockCorpusWindows
from emblema.evaluation.adapters.persistence.downstream_task_repository import (
    SqlAlchemyDownstreamTaskRepository,
)
from emblema.evaluation.adapters.persistence.evaluation_campaign_repository import (
    SqlAlchemyEvaluationCampaignRepository,
)
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.exceptions import UnknownCandidateError
from emblema.shared.adapters.queues.celery_job_queue import CeleryJobQueue
from tests.evaluation.support import LORA, WEIGHTS, adaptation_schedule, boosting, convolutions
from tests.support.settings import unreachable_store


def process(tmp_path: Path) -> CompositionRoot:
    return CompositionRoot(
        unreachable_store(),
        workspace=tmp_path,
        corpora=tmp_path,
        schedule=adaptation_schedule(),
        lora=LORA,
        backbone=WEIGHTS,
        boosting=boosting(),
        convolutions=convolutions(),
    )


def test_without_overrides_the_process_runs_on_what_the_settings_name(tmp_path: Path) -> None:
    root = process(tmp_path)

    assert isinstance(root.adapters.corpus, BlockCorpusWindows)
    assert isinstance(root.adapters.tasks, SqlAlchemyDownstreamTaskRepository)
    assert isinstance(root.adapters.campaigns, SqlAlchemyEvaluationCampaignRepository)
    assert isinstance(root.adapters.jobs, CeleryJobQueue)


def test_a_process_bringing_neither_settings_nor_a_store_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="store"):
        CompositionRoot(
            workspace=tmp_path,
            corpora=tmp_path,
            schedule=adaptation_schedule(),
            lora=LORA,
            backbone=WEIGHTS,
            boosting=boosting(),
            convolutions=convolutions(),
        )


def test_every_registered_candidate_is_reachable_and_nothing_else_is(tmp_path: Path) -> None:
    held = process(tmp_path).adapters.candidates

    for ref in KnownArms.refs():
        assert held.describe(ref).kind is CandidateKind.NEURAL
    for ref in KnownBaselines.refs():
        assert held.describe(ref).kind is CandidateKind.CLASSICAL
    with pytest.raises(UnknownCandidateError):
        held.describe(CandidateRef("absent"))


def test_assembling_this_process_loads_neither_stack_the_grid_is_run_with() -> None:
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
