from dataclasses import dataclass

from emblema.pretraining.application.use_cases.pretrain_backbone import (
    PretrainBackbone,
    PretrainBackboneCommand,
)
from emblema.pretraining.domain.exceptions import PretrainingResultRejectedError
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.pretraining.ports.backbone_repository import BackboneRepository
from emblema.pretraining.ports.handoff_exchange import HandoffExchange
from emblema.pretraining.ports.training_corpus_reader import TrainingCorpusReader
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.clock import Clock


@dataclass(frozen=True, kw_only=True)
class AcceptPretrainingResultCommand:
    """A result to take back from the machine that trained.

    Attributes:
        result: The result to accept; the runtime the use case was given must be reading it.
    """

    result: ArtifactRef


class AcceptPretrainingResult:
    """Takes a result back from the machine that trained and registers the weights it delivered.

    The result is held to the backbone in the registry — which backbone, which code, which
    configuration, which data, which run — and then replayed through the same use case any run
    goes through, with the runtime that reads results: the tracker records it as it would a run
    made here, and the runtime holds the epochs to the configuration and the corpus this process
    reads for itself. What the runtime replayed is then held to the result accepted, because the
    runtime was built on a result of its own and nothing but that comparison says the two are
    one document. Only then is the backbone delivered, with the result beside the weights, and
    saved. A backbone that is already ready refuses before anything is replayed, so a delivery
    taken twice does not record a run twice.
    """

    def __init__(
        self,
        exchange: HandoffExchange,
        reader: TrainingCorpusReader,
        backbones: BackboneRepository,
        pretrain: PretrainBackbone,
        clock: Clock,
    ) -> None:
        self._exchange = exchange
        self._reader = reader
        self._backbones = backbones
        self._pretrain = pretrain
        self._clock = clock

    def __call__(self, command: AcceptPretrainingResultCommand) -> BackboneId:
        """Accept the result and return the backbone it made ready.

        Raises:
            BackboneNotFoundError: If the result is for a backbone the registry does not know.
            BackboneAlreadyDeliveredError: If that backbone already has its weights.
            PretrainingResultRejectedError: If the result is not the delivery the backbone waits
                for, not the run this process reads the order as, or not the result the runtime
                replayed.
            UnreadableHandoffDocumentError: If what the reference names is not a result.
        """
        result = self._exchange.read_result(command.result)
        backbone = self._backbones.get(result.backbone)
        backbone.require_open()
        result.require_delivery_for(backbone)
        corpus = self._reader.read(backbone.input.manifest, backbone.configuration.corpus_share)
        outcome = self._pretrain(
            PretrainBackboneCommand(
                configuration=backbone.configuration,
                corpus=corpus,
                run=backbone.run,
                resume_from=result.resumed_from,
            )
        )
        if outcome != result.outcome:
            raise PretrainingResultRejectedError(
                f"the runtime replayed a run delivering {outcome.backbone.key!r}, "
                f"the result accepted delivers {result.outcome.backbone.key!r}"
            )
        self._backbones.save(backbone.deliver(command.result, outcome.backbone, self._clock.now()))
        return backbone.id
