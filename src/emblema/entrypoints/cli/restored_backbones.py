from torch import nn

from emblema.evaluation.domain.exceptions import UnknownBackboneError
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.training.trained_model import TrainedModel
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore


class RestoredBackbones:
    """Evaluation's encoders out of Pretraining's stored model: the seam the process wires.

    Evaluation asks for an encoder through a shape of call and may not import the module that
    answers it; a process knows both sides, so this is where the trained model Pretraining stored
    becomes the backbone Evaluation adapts. One instance stands for one stored model: the
    pretrained encoder is that model's, a fresh one is the same shape over the same vocabulary
    with weights drawn anew, and a request for other weights is a wiring error rather than a
    second download.
    """

    def __init__(self, store: ArtifactStore, weights: ArtifactRef) -> None:
        """Serve the model stored under ``weights``.

        Raises:
            UnreadableTrainedModelError: If the artifact is not a trained model.
            ArtifactNotFoundError: If nothing is stored under the reference.
            ArtifactIntegrityError: If the stored bytes do not hash to the reference's checksum.
        """
        self._weights = weights
        self._trained = TrainedModel.read(store.get(weights))

    @property
    def width(self) -> int:
        return self._trained.architecture.width

    def pretrained(self, weights: ArtifactRef) -> nn.Module:
        """The stored encoder, in evaluation mode.

        Raises:
            UnknownBackboneError: If other weights are asked for than this was built over.
        """
        if weights != self._weights:
            raise UnknownBackboneError(
                f"this process serves the backbone {self._weights.key}, not {weights.key}"
            )
        return self._trained.build().encoder

    def fresh(self) -> nn.Module:
        return SetEncoder.for_vocabulary(self._trained.architecture, self._trained.vocabulary_size)
