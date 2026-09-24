"""A run of an order: the arguments, the code it is run at, and what it prints.

The process reaches no database and no broker; what is held is that it answers the cells of
the order it is handed, through the candidates it is given, at the revision the machine runs,
and prints the one reference a person carries back.
"""

import subprocess
import sys

import pytest

from emblema.entrypoints.cli.campaign_run.campaign_run_cli import CampaignRunCli
from emblema.entrypoints.cli.campaign_run.composition_root import CompositionRoot
from emblema.entrypoints.source_revision import SourceRevision
from emblema.evaluation.adapters.in_memory.campaign_handoff import InMemoryCampaignHandoff
from emblema.evaluation.adapters.in_memory.candidate_provider import InMemoryCandidateProvider
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.domain.exceptions import CampaignOrderRejectedError
from emblema.evaluation.domain.handoff.campaign_order import CampaignOrder
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.evaluation.support import CAMPAIGN, CONTENDER, CONTROL, campaign, candidate, task, units


class Pinned(SourceRevision):
    """The revision a test says the code is at, whatever the tree around it holds."""

    def __init__(self, commit: str = "abc123") -> None:
        super().__init__()
        self._commit = commit

    def current(self) -> str:
        return self._commit


def placed(handoff: InMemoryCampaignHandoff) -> ArtifactRef:
    grid = campaign()
    return handoff.place(
        CampaignOrder(
            campaign=CAMPAIGN,
            task=task(),
            evaluations=tuple(grid.evaluation_of(cell) for cell in grid.pending()),
            git_commit="abc123",
        )
    )


def root(handoff: InMemoryCampaignHandoff) -> CompositionRoot:
    return CompositionRoot(
        handoff=handoff,
        tasks=InMemoryDownstreamTaskRepository(),
        candidates=InMemoryCandidateProvider(
            (candidate(CONTROL), candidate(CONTENDER)),
            tuple(sorted(units("c0", "c1"), key=str)),
            lambda cell: (1.0, 2.0),
            InMemoryArtifactStore(),
        ),
    )


def argv(ref: ArtifactRef, *more: str) -> list[str]:
    return ["--order", ref.key, str(ref.checksum), *more]


def test_the_run_prints_the_reference_of_every_cell_it_answered() -> None:
    handoff = InMemoryCampaignHandoff()
    order = placed(handoff)
    cli = CampaignRunCli(Pinned())

    printed = cli.execute(cli.parse(argv(order)), root(handoff).services)

    key, checksum = printed.split()
    answered = handoff.read_result(ArtifactRef(key, Checksum.parse(checksum)))
    assert answered.cells == frozenset(handoff.read_order(order).cells)


def test_a_run_resumed_carries_the_reference_it_resumes_from() -> None:
    handoff = InMemoryCampaignHandoff()
    order = placed(handoff)
    earlier = ArtifactRef("durable/earlier", Checksum.of_bytes(b"earlier"))

    invocation = CampaignRunCli(Pinned()).parse(
        argv(order, "--resume", earlier.key, str(earlier.checksum))
    )

    assert invocation.command.resume == earlier
    assert invocation.command.git_commit == "abc123"


def test_a_machine_on_other_code_runs_nothing() -> None:
    handoff = InMemoryCampaignHandoff()
    order = placed(handoff)
    cli = CampaignRunCli(Pinned("def456"))

    with pytest.raises(CampaignOrderRejectedError):
        cli.execute(cli.parse(argv(order)), root(handoff).services)

    assert handoff.reported == []


def test_assembling_this_process_loads_neither_stack_until_an_order_names_one() -> None:
    # Which stack a run carries is decided by the order it reads, and on some platforms the two
    # cannot share a process — so neither may be loaded by the mere import of the root.
    read = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import emblema.entrypoints.cli.campaign_run.composition_root  # noqa: F401\n"
            "print(sorted(m for m in sys.modules if m in {'torch', 'xgboost', 'sklearn'}))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert read.stdout.strip() == "[]"
