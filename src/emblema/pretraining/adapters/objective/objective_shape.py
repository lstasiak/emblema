from dataclasses import dataclass
from typing import Self

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.latent_prediction import LatentPrediction
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.objective.pretraining_objective import PretrainingObjective
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.objective_target import ObjectiveTarget


@dataclass(frozen=True)
class ObjectiveShape:
    """Everything that decides which objective is built and in what shape, without its weights.

    The one place that knows every objective there is: a run builds its objective here and an
    artifact rebuilds it here, so the two cannot drift apart. Dropout is left to the caller — a
    run names its own, and weights being read back are read without it.

    Attributes:
        architecture: Shape of the encoder.
        vocabulary_size: Channels of the corpus the run reads, which sizes the channel table.
        decoder_layers: Blocks of the part that is thrown away: the decoder that predicts a
            value, or the predictor that predicts a state.
        target: What the hidden tokens' predictions are compared with, which picks the objective.
    """

    architecture: EncoderArchitecture
    vocabulary_size: int
    decoder_layers: int
    target: ObjectiveTarget

    @classmethod
    def of(cls, configuration: ExperimentConfiguration, vocabulary_size: int) -> Self:
        return cls(
            architecture=configuration.architecture,
            vocabulary_size=vocabulary_size,
            decoder_layers=configuration.decoder_layers,
            target=configuration.target,
        )

    def build(self, *, dropout: float = 0.0) -> PretrainingObjective:
        """The objective in this shape, its weights drawn from torch's generator as it stands.

        Raises:
            ValueError: If the target reads more blocks than the encoder has.
        """
        encoder = SetEncoder.for_vocabulary(
            self.architecture, self.vocabulary_size, dropout=dropout
        )
        if self.target.is_latent:
            return LatentPrediction(
                encoder, predictor_layers=self.decoder_layers, target=self.target, dropout=dropout
            )
        return MaskedReconstruction(encoder, decoder_layers=self.decoder_layers, dropout=dropout)
