import io
from dataclasses import dataclass
from typing import Any, Self

import torch

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.training.exceptions import (
    UNREADABLE_BYTES,
    UnreadableTrainedModelError,
)
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration


@dataclass(frozen=True)
class TrainedModel:
    """What a finished run keeps: the backbone, the decoder that trained it, and their shape.

    The optimiser's state is left out — it belongs to a run that is over — and the decoder is kept
    even though it is thrown away for transfer, because without it the run's own predictions
    cannot be reproduced, and a diagnostic that scores something other than what was stored is
    scoring the wrong thing. The shape travels with the weights so the artifact rebuilds a model
    on its own, without the configuration that produced it.

    Attributes:
        architecture: Shape of the encoder the weights load into.
        vocabulary_size: Channels of the corpus the run read, which sizes the channel table.
        decoder_layers: Blocks of the decoder.
        weights: State of the whole objective, on the host.
    """

    architecture: EncoderArchitecture
    vocabulary_size: int
    decoder_layers: int
    weights: dict[str, Any]

    @classmethod
    def of(
        cls,
        configuration: ExperimentConfiguration,
        vocabulary_size: int,
        model: MaskedReconstruction,
    ) -> Self:
        """The model as it ended, with its weights copied to the host."""
        return cls(
            architecture=configuration.architecture,
            vocabulary_size=vocabulary_size,
            decoder_layers=configuration.decoder_layers,
            weights={key: value.detach().to("cpu") for key, value in model.state_dict().items()},
        )

    def to_bytes(self) -> bytes:
        """The model as the bytes the artifact store keeps."""
        buffer = io.BytesIO()
        torch.save(
            {
                "width": self.architecture.width,
                "heads": self.architecture.heads,
                "layers": self.architecture.layers,
                "feedforward_width": self.architecture.feedforward_width,
                "time_frequencies": self.architecture.time_frequencies,
                "vocabulary_size": self.vocabulary_size,
                "decoder_layers": self.decoder_layers,
                "weights": self.weights,
            },
            buffer,
        )
        return buffer.getvalue()

    @classmethod
    def read(cls, content: bytes) -> Self:
        """The model those bytes hold.

        Raises:
            UnreadableTrainedModelError: If the bytes are not a trained model of ours.
        """
        try:
            stored = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
            architecture = EncoderArchitecture(
                width=stored["width"],
                heads=stored["heads"],
                layers=stored["layers"],
                feedforward_width=stored["feedforward_width"],
                time_frequencies=stored["time_frequencies"],
            )
            return cls(
                architecture=architecture,
                vocabulary_size=stored["vocabulary_size"],
                decoder_layers=stored["decoder_layers"],
                weights=stored["weights"],
            )
        except UNREADABLE_BYTES as error:
            raise UnreadableTrainedModelError(f"not a trained model this can read: {error}") from (
                error
            )

    def build(self) -> MaskedReconstruction:
        """The objective rebuilt on the host, in evaluation mode, holding these weights.

        Dropout is left at zero: a model being read back is being asked what it predicts, and the
        dropout it trained under is a property of the run, not of the weights.

        Raises:
            UnreadableTrainedModelError: If the weights do not fit the shape stored with them.
        """
        model = MaskedReconstruction(
            SetEncoder.for_vocabulary(self.architecture, self.vocabulary_size),
            decoder_layers=self.decoder_layers,
        )
        try:
            model.load_state_dict(self.weights)
        except RuntimeError as error:
            raise UnreadableTrainedModelError(
                f"weights do not fit the shape stored: {error}"
            ) from (error)
        return model.eval()
