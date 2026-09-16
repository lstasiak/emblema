from dataclasses import dataclass

from emblema.pretraining.application.use_cases.pretrain_backbone import (
    PretrainBackbone,
    PretrainBackboneCommand,
)
from emblema.pretraining.domain.handoff.pretraining_result import PretrainingResult
from emblema.pretraining.ports.handoff_exchange import HandoffExchange
from emblema.pretraining.ports.training_corpus_reader import TrainingCorpusReader
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class FulfilPretrainingOrderCommand:
    """An order to run on this machine.

    Attributes:
        order: The order to fulfil.
        git_commit: Revision of the code this machine runs; the process knows it, the use case
            does not, so it is stated rather than looked up.
        resume_from: Checkpoint of an interrupted run of this order, to pick up from.
    """

    order: ArtifactRef
    git_commit: str
    resume_from: ArtifactRef | None = None


class FulfilPretrainingOrder:
    """Runs an order on this machine and reports the result back through the exchange.

    What the machine that trains does, whichever machine it is: read the order, stop if the code
    running here is not the code ordered, read the corpus the order names and stop if it is not
    the corpus ordered, train through the same use case a run of any kind goes through, and
    report what came out with the facts the acceptance will check. Both stops come before the
    run, because a result the acceptance would refuse is a session of an accelerator wasted.
    """

    def __init__(
        self, exchange: HandoffExchange, reader: TrainingCorpusReader, pretrain: PretrainBackbone
    ) -> None:
        self._exchange = exchange
        self._reader = reader
        self._pretrain = pretrain

    def __call__(self, command: FulfilPretrainingOrderCommand) -> ArtifactRef:
        """Run the order and return the reference to the result.

        Raises:
            PretrainingOrderRejectedError: If this machine runs other code than the order names,
                or the corpus read is not the one the order signed.
            UnreadableHandoffDocumentError: If what the reference names is not an order.
            UnreadablePublishedCorpusError: If the manifest or its block cannot be read.
        """
        order = self._exchange.read_order(command.order)
        order.require_commit(command.git_commit)
        corpus = self._reader.read(order.manifest)
        order.require_read(corpus)
        outcome = self._pretrain(
            PretrainBackboneCommand(
                configuration=order.configuration,
                corpus=corpus,
                run=order.run,
                resume_from=command.resume_from,
            )
        )
        return self._exchange.report(
            PretrainingResult(
                order=command.order,
                backbone=order.backbone,
                configuration=order.configuration,
                corpus=corpus.shape,
                git_commit=command.git_commit,
                resumed_from=command.resume_from,
                outcome=outcome,
            )
        )
