"""What the command line turns its arguments into and prints, without a process assembled."""

import re
from pathlib import Path

import pytest

from emblema.entrypoints.cli.pretrain.pretrain_cli import NO_REGISTRY, PretrainCli
from emblema.entrypoints.cli.pretrain.services import Services
from emblema.entrypoints.source_revision import SourceRevision
from emblema.pretraining.adapters.experiments.experiment_file import ExperimentFile
from emblema.pretraining.adapters.in_memory.experiment_tracker import InMemoryExperimentTracker
from emblema.pretraining.adapters.in_memory.training_runtime import InMemoryTrainingRuntime
from emblema.pretraining.application.use_cases.accept_pretraining_result import (
    AcceptPretrainingResultCommand,
)
from emblema.pretraining.application.use_cases.fulfil_pretraining_order import (
    FulfilPretrainingOrderCommand,
)
from emblema.pretraining.application.use_cases.order_pretraining import OrderPretrainingCommand
from emblema.pretraining.application.use_cases.pretrain_backbone import PretrainBackbone
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.support.handoff import COMMIT, MANIFEST, ORDER_REF
from tests.support.handoff_process import InMemoryHandoff, order_command

EXPERIMENT = Path("experiments/control-a-s.toml")
MIXED_EXPERIMENT = Path("experiments/backbone-mixed-m.toml")
MIXED = ("cmapss", "skab", "smd", "esa_ad")
RESULT = ArtifactRef("durable/result", Checksum.of_bytes(b"result"))
TRACKING = "http://127.0.0.1:5000"


class KnownRevision(SourceRevision):
    """A revision that answers without a package or a tree."""

    def current(self) -> str:
        return COMMIT


class DirtyRevision(SourceRevision):
    """A tree with uncommitted changes."""

    def current(self) -> str:
        return f"{COMMIT}{self.DIRTY}"


def parse(*arguments: str, revision: SourceRevision | None = None):
    return PretrainCli(revision=KnownRevision() if revision is None else revision).parse(
        list(arguments)
    )


def reference(ref: ArtifactRef) -> tuple[str, str]:
    return ref.key, str(ref.checksum)


def order_arguments(*more: str) -> tuple[str, ...]:
    return ("order", "--experiment", str(EXPERIMENT), "--corpus", *reference(MANIFEST), *more)


def test_an_order_is_the_experiment_file_over_the_manifest_at_the_revision_read() -> None:
    invocation = parse(*order_arguments("--run", "r1"))

    command = invocation.command
    assert isinstance(command, OrderPretrainingCommand)
    assert command.configuration == ExperimentFile.load(EXPERIMENT).configuration()
    assert command.corpora == (("control-a", MANIFEST),)
    assert (command.run, command.git_commit) == ("r1", COMMIT)
    assert invocation.workspace == Path("data/artifacts")
    assert (invocation.device, invocation.tracking_uri) == (None, None)


def test_an_order_pairs_the_corpora_the_experiment_names_with_the_manifests_in_order() -> None:
    manifests = [ArtifactRef(f"durable/{name}", Checksum.of_bytes(name.encode())) for name in MIXED]
    invocation = parse(
        "order",
        "--experiment",
        str(MIXED_EXPERIMENT),
        *(argument for ref in manifests for argument in ("--corpus", *reference(ref))),
        "--run",
        "r1",
    )

    command = invocation.command
    assert isinstance(command, OrderPretrainingCommand)
    assert command.corpora == tuple(zip(MIXED, manifests, strict=True))


def test_an_order_with_other_than_one_manifest_per_corpus_is_refused() -> None:
    with pytest.raises(SystemExit, match=r"names 1 corpora \(control-a\); give --corpus once"):
        parse(*order_arguments("--corpus", *reference(ORDER_REF), "--run", "r1"))


def test_a_revision_stated_on_the_command_line_wins_over_the_one_read() -> None:
    invocation = parse(*order_arguments("--run", "r1", "--commit", "abc123"))

    assert isinstance(invocation.command, OrderPretrainingCommand)
    assert invocation.command.git_commit == "abc123"


def test_an_order_from_a_dirty_tree_is_refused_unless_the_revision_is_stated() -> None:
    with pytest.raises(SystemExit, match="uncommitted changes"):
        parse(*order_arguments("--run", "r1"), revision=DirtyRevision())

    stated = parse(*order_arguments("--run", "r1", "--commit", COMMIT), revision=DirtyRevision())
    assert isinstance(stated.command, OrderPretrainingCommand)
    assert stated.command.git_commit == COMMIT


def test_a_run_on_a_dirty_tree_carries_the_mark_for_the_order_to_refuse() -> None:
    invocation = parse("run", "--order", *reference(ORDER_REF), revision=DirtyRevision())

    assert isinstance(invocation.command, FulfilPretrainingOrderCommand)
    assert invocation.command.git_commit == f"{COMMIT}-dirty"


def test_a_run_fulfils_the_order_named_where_it_is_told_to_and_may_be_picked_up() -> None:
    invocation = parse(
        "run",
        "--order",
        *reference(ORDER_REF),
        "--device",
        "cpu",
        "--resume-from",
        *reference(RESULT),
        "--track",
        "sqlite:///runs.db",
        "--num-workers",
        "2",
        "--progress-every",
        "50",
        "--workspace",
        "elsewhere",
    )

    command = invocation.command
    assert isinstance(command, FulfilPretrainingOrderCommand)
    assert (command.order, command.git_commit, command.resume_from) == (ORDER_REF, COMMIT, RESULT)
    assert (invocation.device, invocation.tracking_uri) == ("cpu", "sqlite:///runs.db")
    assert (invocation.num_workers, invocation.workspace) == (2, Path("elsewhere"))
    assert invocation.progress_every == 50


def test_accepting_names_the_result_and_where_to_record_it() -> None:
    invocation = parse("accept", "--result", *reference(RESULT), "--track", TRACKING)

    assert invocation.command == AcceptPretrainingResultCommand(result=RESULT)
    assert invocation.tracking_uri == TRACKING


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["order", "--experiment", str(EXPERIMENT), "--run", "r1"],
        ["run"],
        ["accept", "--result", "only-a-key", "--track", TRACKING],
        ["accept", "--result", *reference(RESULT)],
    ],
    ids=[
        "nothing",
        "order_without_corpus",
        "run_without_order",
        "reference_without_checksum",
        "accept_without_a_tracker",
    ],
)
def test_an_incomplete_invocation_is_refused(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse(*arguments)


def services_of(machines: InMemoryHandoff, result: ArtifactRef | None = None) -> Services:
    return Services(
        pretrain_backbone=PretrainBackbone(
            InMemoryTrainingRuntime(machines.store), InMemoryExperimentTracker()
        ),
        fulfil_pretraining_order=machines.fulfil(),
        order_pretraining=machines.order(),
        accept_pretraining_result=None if result is None else machines.accept(result),
    )


def test_the_three_steps_print_what_the_next_one_is_handed() -> None:
    machines = InMemoryHandoff()
    cli = PretrainCli(revision=KnownRevision())

    ordered = cli.execute(order_command(), services_of(machines))
    backbone, order_key, order_checksum = ordered.splitlines()
    fulfilled = cli.execute(
        FulfilPretrainingOrderCommand(
            order=ArtifactRef(order_key, Checksum.parse(order_checksum)), git_commit=COMMIT
        ),
        services_of(machines),
    )
    result_key, result_checksum = fulfilled.splitlines()
    result = ArtifactRef(result_key, Checksum.parse(result_checksum))
    accepted = cli.execute(
        AcceptPretrainingResultCommand(result=result), services_of(machines, result)
    )

    assert accepted == backbone
    assert machines.backbones.get(BackboneId.parse(backbone)).is_ready


def test_a_process_without_a_registry_refuses_to_order_or_accept_and_says_why() -> None:
    machines = InMemoryHandoff()
    without = Services(
        pretrain_backbone=services_of(machines).pretrain_backbone,
        fulfil_pretraining_order=machines.fulfil(),
        order_pretraining=None,
        accept_pretraining_result=None,
    )
    cli = PretrainCli(revision=KnownRevision())

    with pytest.raises(SystemExit, match=re.escape(NO_REGISTRY)):
        cli.execute(order_command(), without)
    with pytest.raises(SystemExit, match=re.escape(NO_REGISTRY)):
        cli.execute(AcceptPretrainingResultCommand(result=RESULT), without)
