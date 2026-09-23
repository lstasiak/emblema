from pathlib import Path
from typing import Self

from emblema.config.settings import Settings
from emblema.entrypoints.cli.pretrain.adapters import Adapters
from emblema.entrypoints.cli.pretrain.services import Services
from emblema.entrypoints.configured import configured_engine, configured_store, settings_for
from emblema.pretraining.adapters.blocks.block_training_corpus_reader import (
    BlockTrainingCorpusReader,
)
from emblema.pretraining.adapters.handoff.artifact_store_handoff_exchange import (
    ArtifactStoreHandoffExchange,
)
from emblema.pretraining.adapters.handoff.handoff_training_runtime import HandoffTrainingRuntime
from emblema.pretraining.adapters.in_memory.experiment_tracker import InMemoryExperimentTracker
from emblema.pretraining.adapters.mlflow.mlflow_experiment_tracker import MlflowExperimentTracker
from emblema.pretraining.adapters.persistence.backbone_repository import (
    SqlAlchemyBackboneRepository,
)
from emblema.pretraining.adapters.training.torch_training_runtime import TorchTrainingRuntime
from emblema.pretraining.application.use_cases.accept_pretraining_result import (
    AcceptPretrainingResult,
)
from emblema.pretraining.application.use_cases.fulfil_pretraining_order import (
    FulfilPretrainingOrder,
)
from emblema.pretraining.application.use_cases.order_pretraining import OrderPretraining
from emblema.pretraining.application.use_cases.pretrain_backbone import PretrainBackbone
from emblema.pretraining.ports.backbone_repository import BackboneRepository
from emblema.pretraining.ports.experiment_tracker import ExperimentTracker
from emblema.pretraining.ports.handoff_exchange import HandoffExchange
from emblema.pretraining.ports.training_corpus_reader import TrainingCorpusReader
from emblema.pretraining.ports.training_runtime import TrainingRuntime
from emblema.shared.adapters.system.clock import SystemClock
from emblema.shared.adapters.system.id_generator import Uuid4IdGenerator
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.id_generator import IdGenerator


class CompositionRoot:
    """Assembles the pretraining process: every adapter chosen once, every use case built on them.

    One process runs one of three things — orders a run, fulfils an order on this machine, or
    accepts a result — and the runtime is chosen by which: a process given a result to accept
    replays it, any other trains here on the device it names. The tracker is one per process,
    which is one per run. A process whose settings name no database is the machine that only
    trains: it gets no registry and neither of the use cases that write to one.

    Attributes:
        adapters: The port implementations the process runs on.
        services: The use cases, each already holding its dependencies.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        workspace: Path,
        device: str | None = None,
        tracking_uri: str | None = None,
        result: ArtifactRef | None = None,
        num_workers: int = 0,
        progress_every: int = 0,
        store: ArtifactStore | None = None,
        backbones: BackboneRepository | None = None,
        reader: TrainingCorpusReader | None = None,
        exchange: HandoffExchange | None = None,
        runtime: TrainingRuntime | None = None,
        tracker: ExperimentTracker | None = None,
        clock: Clock | None = None,
        ids: IdGenerator | None = None,
    ) -> None:
        """Assemble the process.

        Args:
            settings: Values read from the environment; only the root and the adapters see them.
                Needed only to build the store or the repository; a process bringing both of its
                own is assembled by ``over`` instead, which requires them.
            workspace: Directory blocks are fetched to and mapped from.
            device: Where a run fulfilled here computes; the machine's accelerator unless given.
            tracking_uri: What the run is recorded against; a tracker in memory unless given.
            result: The result this process accepts, if it accepts one: the runtime then replays
                it rather than training.
            num_workers: Processes collating batches for a run fulfilled here.
            progress_every: Optimiser steps between two progress lines of a run fulfilled here;
                zero for none.
            store: Artifact store; the configured S3-compatible bucket unless given.
            backbones: Registry of backbones; the configured metadata database unless given, and
                none where the settings name no database.
            reader: Reader of published corpora; the block reader over the store unless given.
            exchange: Where orders and results pass; documents in the store unless given.
            runtime: What trains, or replays; chosen by ``result`` and ``device`` unless given.
            tracker: What records the run; chosen by ``tracking_uri`` unless given.
            clock: Source of the current instant; the system clock unless given.
            ids: Source of new identifiers; random UUIDs unless given.

        Raises:
            ValueError: If the store or the repository is left to the root without settings to
                build it from.
        """
        chosen_store = configured_store(settings_for(settings, "store")) if store is None else store
        chosen_exchange = (
            ArtifactStoreHandoffExchange(chosen_store) if exchange is None else exchange
        )
        self.adapters = Adapters(
            store=chosen_store,
            backbones=(
                self._registry(settings_for(settings, "backbones"))
                if backbones is None
                else backbones
            ),
            reader=BlockTrainingCorpusReader(chosen_store, workspace) if reader is None else reader,
            exchange=chosen_exchange,
            runtime=(
                self._runtime(
                    chosen_store, chosen_exchange, device, result, num_workers, progress_every
                )
                if runtime is None
                else runtime
            ),
            tracker=self._tracker(tracking_uri) if tracker is None else tracker,
            clock=SystemClock() if clock is None else clock,
            ids=Uuid4IdGenerator() if ids is None else ids,
        )
        self.services = self._services(self.adapters)

    @classmethod
    def over(
        cls,
        *,
        store: ArtifactStore,
        backbones: BackboneRepository,
        workspace: Path,
        device: str | None = None,
        tracking_uri: str | None = None,
        result: ArtifactRef | None = None,
        num_workers: int = 0,
        progress_every: int = 0,
        reader: TrainingCorpusReader | None = None,
        exchange: HandoffExchange | None = None,
        runtime: TrainingRuntime | None = None,
        tracker: ExperimentTracker | None = None,
        clock: Clock | None = None,
        ids: IdGenerator | None = None,
    ) -> Self:
        """The process over a store and a registry of its own, nothing read from the environment.

        Those two are the only adapters the root builds from settings, and here they are required:
        what the general constructor refuses at runtime when settings are missing, this refuses at
        the type.
        """
        return cls(
            workspace=workspace,
            device=device,
            tracking_uri=tracking_uri,
            result=result,
            num_workers=num_workers,
            progress_every=progress_every,
            store=store,
            backbones=backbones,
            reader=reader,
            exchange=exchange,
            runtime=runtime,
            tracker=tracker,
            clock=clock,
            ids=ids,
        )

    @staticmethod
    def _registry(settings: Settings) -> BackboneRepository | None:
        if settings.database is None:
            return None
        return SqlAlchemyBackboneRepository(configured_engine(settings))

    @staticmethod
    def _runtime(
        store: ArtifactStore,
        exchange: HandoffExchange,
        device: str | None,
        result: ArtifactRef | None,
        num_workers: int,
        progress_every: int,
    ) -> TrainingRuntime:
        if result is not None:
            return HandoffTrainingRuntime(exchange, store, result)
        return TorchTrainingRuntime(
            store, device=device, num_workers=num_workers, progress_every=progress_every
        )

    @staticmethod
    def _tracker(tracking_uri: str | None) -> ExperimentTracker:
        return (
            InMemoryExperimentTracker()
            if tracking_uri is None
            else MlflowExperimentTracker(tracking_uri)
        )

    @staticmethod
    def _services(adapters: Adapters) -> Services:
        pretrain = PretrainBackbone(adapters.runtime, adapters.tracker)
        backbones = adapters.backbones
        return Services(
            pretrain_backbone=pretrain,
            fulfil_pretraining_order=FulfilPretrainingOrder(
                adapters.exchange, adapters.reader, pretrain
            ),
            order_pretraining=(
                None
                if backbones is None
                else OrderPretraining(
                    adapters.reader, backbones, adapters.exchange, adapters.ids, adapters.clock
                )
            ),
            accept_pretraining_result=(
                None
                if backbones is None
                else AcceptPretrainingResult(
                    adapters.exchange, adapters.reader, backbones, pretrain, adapters.clock
                )
            ),
        )
