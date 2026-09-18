from collections.abc import Iterator

from emblema.pretraining.domain.exceptions import PretrainingResultRejectedError
from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.training_mixture import TrainingMixture
from emblema.pretraining.ports.handoff_exchange import HandoffExchange
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore


class HandoffTrainingRuntime:
    """A run made on a machine this one only handed work to, replayed from what it reported.

    The third way to fulfil the port: not training on a device here and not counting epochs in
    memory, but reading what another machine trained and reporting it epoch by epoch as though
    it were happening now. The result is held to the run asked for — the configuration, the data
    and the checkpoint, each named when it differs — and to the store, which must hold the
    weights the result names. What passes is what the caller would have seen had the run
    happened here, so the same use case records it and the same tracker keeps the curve.

    Refusals happen when the run is asked for, not when its first epoch is: a caller that
    records runs must not open one for a result that is not the run it ordered.
    """

    def __init__(
        self, exchange: HandoffExchange, store: ArtifactStore, result: ArtifactRef
    ) -> None:
        """Replay the result stored under ``result``, whose weights ``store`` must hold."""
        self._exchange = exchange
        self._store = store
        self._result = result

    def train(
        self,
        configuration: ExperimentConfiguration,
        mixture: TrainingMixture,
        resume_from: ArtifactRef | None = None,
    ) -> Iterator[EpochOutcome]:
        """The epochs the other machine reported, once the result is held to this run.

        Raises:
            PretrainingResultRejectedError: If the result is of another configuration, other
                data or another checkpoint, or names weights the store does not hold.
            UnreadableHandoffDocumentError: If what is stored is not a result.
            ArtifactNotFoundError: If the result is not in the store.
        """
        result = self._exchange.read_result(self._result)
        result.require_run_of(configuration, mixture.shape, resume_from)
        if not self._store.exists(result.outcome.backbone):
            raise PretrainingResultRejectedError(
                f"the result names weights under {result.outcome.backbone.key!r} "
                f"that the store does not hold"
            )
        return iter(result.outcome.epochs)
