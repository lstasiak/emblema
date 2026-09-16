from dataclasses import dataclass

from emblema.pretraining.domain.backbone.backbone import Backbone
from emblema.pretraining.domain.exceptions import PretrainingOrderRejectedError
from emblema.pretraining.domain.handoff.pretraining_order import PretrainingOrder
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.run_signature import RunSignature
from emblema.pretraining.ports.backbone_repository import BackboneRepository
from emblema.pretraining.ports.handoff_exchange import HandoffExchange
from emblema.pretraining.ports.training_corpus_reader import TrainingCorpusReader
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.id_generator import IdGenerator


@dataclass(frozen=True, kw_only=True)
class OrderPretrainingCommand:
    """One run to be made, on this machine or another: what, on which corpus, by which code.

    Attributes:
        configuration: The experiment to run.
        corpus: The corpus the experiment names, which the manifest must publish.
        manifest: The published corpus to train on.
        run: What this run is called within the experiment.
        git_commit: Revision of the code the run is to be made with.
    """

    configuration: ExperimentConfiguration
    corpus: str
    manifest: ArtifactRef
    run: str
    git_commit: str


@dataclass(frozen=True, kw_only=True)
class PlacedPretrainingOrder:
    """What ordering leaves behind: the backbone registered and the order the trainer is handed.

    Attributes:
        backbone: The backbone, registered as ordered.
        order: Reference to the order in the exchange.
    """

    backbone: BackboneId
    order: ArtifactRef


class OrderPretraining:
    """Registers a backbone as ordered and places the order the machine that trains is handed.

    The corpus is read here rather than only described: the signature the result will be held
    to covers how much of the corpus a run reads, and that is known from the windows, not from
    the manifest. The backbone is saved before the order is placed, because an order in the
    exchange for a backbone the registry does not know is a delivery nobody could accept.
    """

    def __init__(
        self,
        reader: TrainingCorpusReader,
        backbones: BackboneRepository,
        exchange: HandoffExchange,
        ids: IdGenerator,
        clock: Clock,
    ) -> None:
        self._reader = reader
        self._backbones = backbones
        self._exchange = exchange
        self._ids = ids
        self._clock = clock

    def __call__(self, command: OrderPretrainingCommand) -> PlacedPretrainingOrder:
        """Register the backbone and place its order.

        Raises:
            PretrainingOrderRejectedError: If the manifest publishes a corpus other than the one
                the experiment names.
            UnreadablePublishedCorpusError: If the manifest or its block cannot be read.
        """
        described = self._reader.describe(command.manifest)
        if described.corpus != command.corpus:
            raise PretrainingOrderRejectedError(
                f"the experiment names corpus {command.corpus!r}, "
                f"the manifest publishes {described.corpus!r}"
            )
        corpus = self._reader.read(command.manifest)
        backbone = Backbone(
            id=self._ids.generate(BackboneId),
            configuration=command.configuration,
            input=described,
            run=command.run,
            git_commit=command.git_commit,
            signature=RunSignature.of(command.configuration, corpus),
            ordered_at=self._clock.now(),
        )
        self._backbones.save(backbone)
        return PlacedPretrainingOrder(
            backbone=backbone.id, order=self._exchange.place(PretrainingOrder.of(backbone))
        )
