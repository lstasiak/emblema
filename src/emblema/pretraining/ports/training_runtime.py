from collections.abc import Iterator
from typing import Protocol

from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.shared.kernel.artifacts import ArtifactRef


class TrainingRuntime(Protocol):
    """Runs a configured experiment over a corpus and reports each epoch as it finishes.

    Where the training happens is the adapter's business: in this process on whatever device it
    finds, or on a platform this one only hands work to. What the port fixes is that a run is
    described by a configuration and a corpus, that it reports epochs one at a time rather than a
    verdict at the end, and that it can be picked up from state it wrote earlier. Reporting per
    epoch is what lets the application log a run while it runs, which is the difference between a
    session that drops after two hours and two hours lost.

    Weights never cross this port: a checkpoint and the backbone are references into the artifact
    store, so a runtime training elsewhere reports the same things a local one does.
    """

    def train(
        self,
        configuration: ExperimentConfiguration,
        corpus: TrainingCorpus,
        resume_from: ArtifactRef | None = None,
    ) -> Iterator[EpochOutcome]:
        """Train ``configuration`` over ``corpus``, yielding one outcome per epoch it runs.

        A run resumed from a checkpoint yields the epochs that were left, numbered as they were
        in the interrupted run, and picks up inside the epoch the checkpoint was written in: the
        windows already seen there are not seen twice. The last epoch of the run carries the
        backbone; the epochs before it carry whatever checkpoints the policy asked for.

        Nothing is trained until the first outcome is asked for, and a caller that stops consuming
        stops the run.

        Raises:
            IncompatibleCheckpointError: If the checkpoint is not one this configuration and
                corpus produced, or the run it comes from has already finished.
            UnsupportedPrecisionError: If the device cannot compute at the declared precision.
            ArtifactNotFoundError: If the checkpoint is not in the store.
            ArtifactIntegrityError: If the stored checkpoint does not hash to its checksum.
        """
        ...
