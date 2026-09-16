"""The pretraining process assembled and run end to end, on adapters that reach nothing outside.

What is checked is the wiring: that three processes built by the root — one that orders, one
that fulfils, one that accepts — make one backbone between them over the adapters they were
given, and that a process without overrides gets the adapters the settings name.
"""

from pathlib import Path
from typing import Any

import pytest

from emblema.config.settings import Settings
from emblema.entrypoints.cli.pretrain.composition_root import CompositionRoot
from emblema.pretraining.adapters.blocks.block_training_corpus_reader import (
    BlockTrainingCorpusReader,
)
from emblema.pretraining.adapters.handoff.artifact_store_handoff_exchange import (
    ArtifactStoreHandoffExchange,
)
from emblema.pretraining.adapters.handoff.handoff_training_runtime import HandoffTrainingRuntime
from emblema.pretraining.adapters.in_memory.backbone_repository import InMemoryBackboneRepository
from emblema.pretraining.adapters.in_memory.experiment_tracker import InMemoryExperimentTracker
from emblema.pretraining.adapters.in_memory.handoff_exchange import InMemoryHandoffExchange
from emblema.pretraining.adapters.in_memory.training_corpus_reader import (
    InMemoryTrainingCorpusReader,
)
from emblema.pretraining.adapters.in_memory.training_runtime import InMemoryTrainingRuntime
from emblema.pretraining.adapters.mlflow.mlflow_experiment_tracker import MlflowExperimentTracker
from emblema.pretraining.adapters.persistence.backbone_repository import (
    SqlAlchemyBackboneRepository,
)
from emblema.pretraining.adapters.training.torch_training_runtime import TorchTrainingRuntime
from emblema.pretraining.application.use_cases.accept_pretraining_result import (
    AcceptPretrainingResult,
    AcceptPretrainingResultCommand,
)
from emblema.pretraining.application.use_cases.fulfil_pretraining_order import (
    FulfilPretrainingOrderCommand,
)
from emblema.pretraining.application.use_cases.order_pretraining import OrderPretraining
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.storage.s3 import S3ArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from tests.support.handoff import COMMIT, CORPUS, MANIFEST, pretraining_input
from tests.support.handoff_process import order_command
from tests.support.settings import ARTIFACT_STORE, unreachable_store

pytestmark = pytest.mark.ml


class Machines:
    """What both machines share, and a process on either of them over it."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.store = InMemoryArtifactStore()
        self.exchange = InMemoryHandoffExchange()
        self.backbones = InMemoryBackboneRepository()
        self.reader = InMemoryTrainingCorpusReader()
        self.reader.publish(MANIFEST, pretraining_input(), CORPUS)

    def process(self, result: ArtifactRef | None = None, **overrides: Any) -> CompositionRoot:
        return CompositionRoot(
            unreachable_store(),
            workspace=self.workspace,
            result=result,
            store=self.store,
            backbones=self.backbones,
            reader=self.reader,
            exchange=self.exchange,
            **overrides,
        )


@pytest.fixture
def machines(tmp_path: Path) -> Machines:
    return Machines(tmp_path / "workspace")


def ordering(root: CompositionRoot) -> OrderPretraining:
    assert root.services.order_pretraining is not None, "a process with a registry orders"
    return root.services.order_pretraining


def accepting(root: CompositionRoot) -> AcceptPretrainingResult:
    assert root.services.accept_pretraining_result is not None, "a process with a registry accepts"
    return root.services.accept_pretraining_result


def test_three_processes_make_one_backbone_over_the_adapters_they_share(
    machines: Machines,
) -> None:
    placed = ordering(machines.process())(order_command())
    reported = machines.process(
        runtime=InMemoryTrainingRuntime(machines.store)
    ).services.fulfil_pretraining_order(
        FulfilPretrainingOrderCommand(order=placed.order, git_commit=COMMIT)
    )
    accepted = accepting(machines.process(result=reported))(
        AcceptPretrainingResultCommand(result=reported)
    )

    ready = machines.backbones.get(accepted)
    assert accepted == placed.backbone
    assert ready.is_ready
    assert ready.artifact is not None
    assert machines.store.exists(ready.artifact)


def test_a_process_accepting_a_result_replays_it_and_records_it(machines: Machines) -> None:
    placed = ordering(machines.process())(order_command())
    reported = machines.process(
        runtime=InMemoryTrainingRuntime(machines.store)
    ).services.fulfil_pretraining_order(
        FulfilPretrainingOrderCommand(order=placed.order, git_commit=COMMIT)
    )
    tracker = InMemoryExperimentTracker()
    replaying = machines.process(result=reported, tracker=tracker)

    assert isinstance(replaying.adapters.runtime, HandoffTrainingRuntime)
    accepting(replaying)(AcceptPretrainingResultCommand(result=reported))
    assert tracker.outcome == machines.exchange.read_result(reported).outcome


def test_a_process_without_a_result_trains_here_on_the_device_named(machines: Machines) -> None:
    training = machines.process(device="cpu")

    assert isinstance(training.adapters.runtime, TorchTrainingRuntime)
    assert training.adapters.runtime.device == "cpu"
    assert isinstance(training.adapters.tracker, InMemoryExperimentTracker)


def test_a_tracking_uri_puts_the_run_on_mlflow(machines: Machines, tmp_path: Path) -> None:
    tracked = machines.process(device="cpu", tracking_uri=f"sqlite:///{tmp_path / 'runs.db'}")

    assert isinstance(tracked.adapters.tracker, MlflowExperimentTracker)


def test_without_overrides_the_process_runs_on_what_the_settings_name(tmp_path: Path) -> None:
    # The overrides above are what every other test here uses, so nothing would otherwise
    # exercise the wiring a real run gets: the bucket and the database the environment names,
    # the exchange and the block reader over that bucket.
    root = CompositionRoot(unreachable_store(), workspace=tmp_path, device="cpu")

    assert isinstance(root.adapters.store, S3ArtifactStore)
    assert isinstance(root.adapters.backbones, SqlAlchemyBackboneRepository)
    assert isinstance(root.adapters.reader, BlockTrainingCorpusReader)
    assert isinstance(root.adapters.exchange, ArtifactStoreHandoffExchange)


def test_a_process_bringing_neither_settings_nor_a_store_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs store given"):
        CompositionRoot(workspace=tmp_path, backbones=InMemoryBackboneRepository())


def test_over_assembles_the_process_on_a_store_and_a_registry_of_its_own(tmp_path: Path) -> None:
    root = CompositionRoot.over(
        store=InMemoryArtifactStore(),
        backbones=InMemoryBackboneRepository(),
        workspace=tmp_path,
        device="cpu",
    )

    assert isinstance(root.adapters.reader, BlockTrainingCorpusReader)
    assert isinstance(root.adapters.runtime, TorchTrainingRuntime)


def test_a_process_without_a_database_trains_and_reports_but_neither_orders_nor_accepts(
    tmp_path: Path,
) -> None:
    # The machine that only trains: a notebook with the bucket's credentials and nothing else.
    root = CompositionRoot(
        Settings(artifact_store=ARTIFACT_STORE, database=None), workspace=tmp_path, device="cpu"
    )

    assert root.adapters.backbones is None
    assert root.services.order_pretraining is None
    assert root.services.accept_pretraining_result is None
    assert root.services.fulfil_pretraining_order is not None
