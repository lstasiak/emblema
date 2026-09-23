"""The three invocations a comparison is declared and handed out by.

None of them runs a cell: a command line that did would be a second way of producing the same
numbers, under whatever the machine at the keyboard happened to be configured with. What is
held here is that each invocation reaches its use case with what the arguments and the file say
and nothing else, and that a name nothing knows is refused before anything is stored.
"""

from collections.abc import Mapping
from pathlib import Path

import pytest

from emblema.entrypoints.cli.campaign.adapters import Adapters
from emblema.entrypoints.cli.campaign.campaign_cli import CampaignCli
from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
from emblema.entrypoints.cli.campaign.services import Services
from emblema.evaluation.adapters.in_memory.candidate_provider import InMemoryCandidateProvider
from emblema.evaluation.adapters.in_memory.corpus_windows import InMemoryCorpusWindows
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.application.use_cases.advance_campaign import (
    RUN_CAMPAIGN_CELL,
    AdvanceCampaign,
)
from emblema.evaluation.application.use_cases.define_campaign import DefineCampaign
from emblema.evaluation.application.use_cases.define_downstream_task import DefineDownstreamTask
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef, TaskId
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.task.corpus_sides import CorpusSides
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.adapters.queues.immediate_job_queue import ImmediateJobQueue
from emblema.shared.jobs.job_argument import JobArgument
from tests.evaluation.support import MANIFEST, OPENED_AT, candidate

CONTROL, TREES = CandidateRef("from_scratch"), CandidateRef("boosted_trees_per_channel")
ENGINES = tuple(f"FD001/{engine}" for engine in range(1, 9))
DECLARED = f"""
name = "baselines-fd001"
tier = "S"

[candidates]
competing = ["{CONTROL}", "{TREES}"]
control = "{CONTROL}"
endpoint = "{TREES}"

[budgets]
labels = ["50", "200"]
endpoint = "200"
seeds = [1, 2]

[rules]
minimum_relative_reduction = 0.1
floor_share = 0.02
secondary_family_size = 8
"""


class Process:
    """The command line over adapters that touch nothing, wired as the real root wires it."""

    def __init__(self) -> None:
        self.submitted: list[Mapping[str, JobArgument]] = []
        tasks = InMemoryDownstreamTaskRepository()
        campaigns = InMemoryEvaluationCampaignRepository()
        corpus = InMemoryCorpusWindows(
            CorpusSides(
                corpus="cmapss",
                training=frozenset(UnitKey(name) for name in ENGINES[:6]),
                validation=frozenset(UnitKey(name) for name in ENGINES[6:]),
            ),
            {UnitKey(name): [60.0, 120.0] for name in ENGINES},
            MANIFEST,
        )
        jobs = ImmediateJobQueue({RUN_CAMPAIGN_CELL: self.submitted.append})
        ids = SequentialIdGenerator()
        self.adapters = Adapters(corpus=corpus, tasks=tasks, campaigns=campaigns, jobs=jobs)
        self.services = Services(
            define_downstream_task=DefineDownstreamTask(tasks, corpus, ids),
            define_campaign=DefineCampaign(
                tasks,
                campaigns,
                InMemoryCandidateProvider((candidate(CONTROL), candidate(TREES)), (), lambda _: ()),
                ids,
                FixedClock(OPENED_AT),
            ),
            advance_campaign=AdvanceCampaign(campaigns, jobs),
        )

    def run(self, *argv: str) -> str:
        return CampaignCli().execute(CampaignCli().parse(argv), self.adapters, self.services)


@pytest.fixture
def process() -> Process:
    return Process()


def defined(process: Process) -> str:
    return process.run(
        "define-task",
        "--corpus",
        MANIFEST.key,
        str(MANIFEST.checksum),
        "--task",
        KnownTasks.TURBOFAN_FD001.name,
    )


def test_a_task_is_cut_out_of_the_corpus_the_invocation_names(process: Process) -> None:
    printed = defined(process)

    stored = process.adapters.tasks.get(TaskId.parse(printed))
    assert stored.corpus == "cmapss"
    assert {str(unit) for unit in stored.tuning_units} <= set(ENGINES)


def test_the_frozen_side_is_the_official_test_set_and_is_learnt_from_by_nobody(
    process: Process,
) -> None:
    stored = process.adapters.tasks.get(TaskId.parse(defined(process)))

    assert stored.split.test.source == "cmapss/test/FD001"
    assert not (stored.tuning_units | stored.validation_units) & stored.split.test.units


def test_a_campaign_is_declared_from_the_file_and_nothing_else(
    process: Process, tmp_path: Path
) -> None:
    task_id = defined(process)
    path = tmp_path / "campaign.toml"
    path.write_text(DECLARED, encoding="utf-8")

    printed = process.run("define", "--file", str(path), "--task", task_id)

    stored = process.adapters.campaigns.get(CampaignId.parse(printed))
    assert [str(named.ref) for named in stored.design.candidates] == [str(CONTROL), str(TREES)]
    assert stored.design.seeds == (1, 2)
    assert stored.task == TaskId.parse(task_id)


def test_advancing_hands_over_every_cell_that_has_not_run(process: Process, tmp_path: Path) -> None:
    task_id = defined(process)
    path = tmp_path / "campaign.toml"
    path.write_text(DECLARED, encoding="utf-8")
    campaign_id = process.run("define", "--file", str(path), "--task", task_id)

    printed = process.run("advance", "--campaign", campaign_id)

    # Two candidates, two budgets, two seeds.
    assert printed == "8"
    assert len(process.submitted) == 8


def test_a_task_nothing_is_registered_under_is_refused_before_anything_is_stored(
    process: Process,
) -> None:
    with pytest.raises(SystemExit, match="no task called"):
        CampaignCli().execute(
            CampaignCli().parse(
                ["define-task", "--corpus", MANIFEST.key, str(MANIFEST.checksum), "--task", "x"]
            ),
            process.adapters,
            process.services,
        )
