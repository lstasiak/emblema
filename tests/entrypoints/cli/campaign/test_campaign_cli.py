"""How a comparison is declared, handed out, announced, ordered out and accepted back.

None of them runs a cell: a command line that did would be another way of producing the same
numbers, under whatever the machine at the keyboard happened to be configured with. What is
held here is that each invocation reaches its use case with what the arguments and the file say
and nothing else, and that a name nothing knows is refused before anything is stored.
"""

import tomllib
from collections.abc import Mapping
from pathlib import Path

import pytest

from emblema.entrypoints.cli.campaign.adapters import Adapters
from emblema.entrypoints.cli.campaign.campaign_cli import CampaignCli
from emblema.entrypoints.cli.campaign.campaign_invocation import CampaignInvocation
from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
from emblema.entrypoints.cli.campaign.services import Services
from emblema.entrypoints.source_revision import SourceRevision
from emblema.evaluation.adapters.in_memory.campaign_handoff import InMemoryCampaignHandoff
from emblema.evaluation.adapters.in_memory.candidate_provider import InMemoryCandidateProvider
from emblema.evaluation.adapters.in_memory.corpus_windows import InMemoryCorpusWindows
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.application.assemblers.campaign_completed_assembler import (
    CampaignCompletedAssembler,
)
from emblema.evaluation.application.use_cases.accept_campaign_order_result import (
    AcceptCampaignOrderResult,
)
from emblema.evaluation.application.use_cases.advance_campaign import (
    RUN_CAMPAIGN_CELL,
    AdvanceCampaign,
)
from emblema.evaluation.application.use_cases.announce_campaign import AnnounceCampaign
from emblema.evaluation.application.use_cases.complete_campaign import CompleteCampaign
from emblema.evaluation.application.use_cases.define_campaign import DefineCampaign
from emblema.evaluation.application.use_cases.define_downstream_task import DefineDownstreamTask
from emblema.evaluation.application.use_cases.order_campaign_cells import OrderCampaignCells
from emblema.evaluation.application.use_cases.record_cell_result import RecordCellResult
from emblema.evaluation.application.use_cases.select_tuned_variants import SelectTunedVariants
from emblema.evaluation.contracts.events import CampaignCompleted
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef, TaskId
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.handoff.campaign_order_result import CampaignOrderResult
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.scoring.unit_error import UnitError
from emblema.evaluation.domain.task.corpus_sides import CorpusSides
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.adapters.queues.immediate_job_queue import ImmediateJobQueue
from emblema.shared.jobs.job_argument import JobArgument
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.evaluation.support import (
    CAMPAIGN,
    FINER,
    MANIFEST,
    OPENED_AT,
    ROCKET,
    SELECTED_BY,
    artifact,
    campaign,
    candidate,
    closed_campaign,
    selection,
)

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
        ids, clock = SequentialIdGenerator(), FixedClock(OPENED_AT)
        self.announced: list[CampaignCompleted] = []
        subscriptions = InMemoryEventSubscriber()
        subscriptions.subscribe(CampaignCompleted, self.announced.append)
        events = InMemoryEventPublisher(subscriptions)
        self.handoff = InMemoryCampaignHandoff()
        outcomes = CampaignCompletedAssembler()
        complete = CompleteCampaign(campaigns, outcomes, clock, ids, events)
        catalogue = InMemoryCandidateProvider(
            (candidate(CONTROL), candidate(TREES)), (), lambda _: ()
        )
        self.adapters = Adapters(
            corpus=corpus,
            candidates=catalogue,
            tasks=tasks,
            campaigns=campaigns,
            jobs=jobs,
            handoff=self.handoff,
        )
        self.services = Services(
            define_downstream_task=DefineDownstreamTask(tasks, corpus, ids),
            define_campaign=DefineCampaign(tasks, campaigns, catalogue, ids, clock),
            advance_campaign=AdvanceCampaign(campaigns, jobs, complete),
            announce_campaign=AnnounceCampaign(campaigns, outcomes, ids, events),
            order_campaign_cells=OrderCampaignCells(campaigns, tasks, self.handoff),
            accept_campaign_order_result=AcceptCampaignOrderResult(
                self.handoff, RecordCellResult(campaigns, complete)
            ),
            select_tuned_variants=SelectTunedVariants(campaigns),
        )

    def run(self, *argv: str) -> str:
        cli = CampaignCli(Pinned())
        return cli.execute(cli.parse(argv), self.adapters, self.services)


class Pinned(SourceRevision):
    """The revision a test says the code is at, whatever the tree around it holds."""

    def current(self) -> str:
        return "abc123"


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


def test_announcing_a_closed_campaign_publishes_it_again_and_prints_what_it_kept(
    process: Process,
) -> None:
    kept = artifact("contender")
    process.adapters.campaigns.save(closed_campaign(kept), seen=0)

    printed = process.run("announce", "--campaign", str(CAMPAIGN))

    assert printed == str(kept.checksum)
    assert [event.campaign for event in process.announced] == [CAMPAIGN]


def test_announcing_a_campaign_still_running_is_refused_with_the_reason(
    process: Process,
) -> None:
    process.adapters.campaigns.save(campaign(), seen=0)

    with pytest.raises(SystemExit, match="has not finished"):
        process.run("announce", "--campaign", str(CAMPAIGN))

    assert process.announced == []


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


@pytest.mark.parametrize(
    ("invocation", "complaint"),
    [
        (CampaignInvocation(what="define"), "give --task"),
        (CampaignInvocation(what="advance"), "give --campaign"),
        (CampaignInvocation(what="announce"), "give --campaign"),
    ],
)
def test_an_invocation_missing_what_it_acts_on_is_refused(
    process: Process, invocation: CampaignInvocation, complaint: str
) -> None:
    # The parser demands both, so this is what happens when something other than the parser
    # builds an invocation — which is the only way `execute` is reached in a test.
    with pytest.raises(SystemExit, match=complaint):
        CampaignCli().execute(invocation, process.adapters, process.services)


def test_an_invocation_asking_for_something_else_is_refused_rather_than_advanced(
    process: Process,
) -> None:
    # The parser admits four names, so this too is what happens when something other than the
    # parser builds an invocation. Answering it as the last branch would hand out a grid's cells
    # for a subcommand nobody wrote.
    with pytest.raises(SystemExit, match="does not 'publish'"):
        CampaignCli().execute(
            CampaignInvocation(what="publish"), process.adapters, process.services
        )

    assert process.submitted == []


def declared(process: Process, tmp_path: Path) -> str:
    path = tmp_path / "campaign.toml"
    path.write_text(DECLARED, encoding="utf-8")
    return process.run("define", "--file", str(path), "--task", defined(process))


def test_an_order_carries_the_outstanding_cells_of_its_pool_at_the_code_this_tree_is_at(
    process: Process, tmp_path: Path
) -> None:
    campaign_id = declared(process, tmp_path)

    key, checksum = process.run("order", "--campaign", campaign_id, "--pool", "ml").split()

    order = process.handoff.read_order(ArtifactRef(key, Checksum.parse(checksum)))
    assert len(order.cells) == 8
    assert order.git_commit == "abc123"
    assert process.submitted == []


def test_accepting_a_result_records_the_cells_it_answered(process: Process, tmp_path: Path) -> None:
    campaign_id = CampaignId.parse(declared(process, tmp_path))
    key, checksum = process.run("order", "--campaign", str(campaign_id), "--pool", "ml").split()
    order = ArtifactRef(key, Checksum.parse(checksum))
    first = process.handoff.read_order(order).cells[0]
    reported = process.handoff.report(
        CampaignOrderResult(
            order=order,
            campaign=campaign_id,
            git_commit="abc123",
            results=(
                CellResult(
                    cell=first,
                    errors=(UnitError(unit=UnitKey("FD001/7"), squared_error=4.0, windows=2),),
                    seconds=1.0,
                    artifact=None,
                ),
            ),
        )
    )

    printed = process.run("accept", "--result", reported.key, str(reported.checksum))

    assert printed == "1"
    assert [r.cell for r in process.adapters.campaigns.get(campaign_id).results] == [first]


def test_an_order_names_a_pool_the_parser_knows() -> None:
    with pytest.raises(SystemExit):
        CampaignCli(Pinned()).parse(["order", "--campaign", "x", "--pool", "gpu"])


def test_select_prints_each_choice_as_the_table_a_comparison_names_it_in(
    process: Process,
) -> None:
    process.adapters.campaigns.save(selection(), seen=0)

    printed = process.run("select", "--campaign", str(SELECTED_BY), "--candidate", str(ROCKET))

    parsed = tomllib.loads(printed)["tuned"]
    assert [entry["budget"] for entry in parsed] == ["50", "200"]
    assert {entry["variant"] for entry in parsed} == {str(FINER)}
    assert {entry["selected_by"] for entry in parsed} == {str(SELECTED_BY)}
