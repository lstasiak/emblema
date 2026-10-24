from dataclasses import dataclass, field
from typing import Self

from torch import nn

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.cell_reconstruction import CellReconstruction
from emblema.pretraining.adapters.objective.contrastive_reconstruction import (
    ContrastiveReconstruction,
)
from emblema.pretraining.adapters.objective.latent_prediction import LatentPrediction
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.objective.pretraining_objective import PretrainingObjective
from emblema.pretraining.adapters.objective.summary_reconstruction import SummaryReconstruction
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.training.contrastive_term import ContrastiveTerm
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.objective_target import ObjectiveTarget
from emblema.pretraining.domain.training.observation_term import ObservationTerm
from emblema.pretraining.domain.training.summary_term import SummaryTerm


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
        contrast: The term beside the value objective that tells a window's halves apart, where
            stated; none picks the value objective alone.
        summary: The term beside the value objective that asks for each channel's level and
            extremes of the hidden part, where stated.
        observation: The terms beside the values of the hidden cells that ask whether and how
            often the channel was measured there, where stated; only over cells.
    """

    architecture: EncoderArchitecture
    vocabulary_size: int
    decoder_layers: int
    target: ObjectiveTarget
    contrast: ContrastiveTerm = field(default_factory=ContrastiveTerm.none)
    summary: SummaryTerm = field(default_factory=SummaryTerm.none)
    observation: ObservationTerm = field(default_factory=ObservationTerm.none)

    @classmethod
    def of(cls, configuration: ExperimentConfiguration, vocabulary_size: int) -> Self:
        return cls(
            architecture=configuration.architecture,
            vocabulary_size=vocabulary_size,
            decoder_layers=configuration.decoder_layers,
            target=configuration.target,
            contrast=configuration.contrast,
            summary=configuration.summary,
            observation=configuration.observation,
        )

    def build(self, *, dropout: float = 0.0) -> PretrainingObjective:
        """The objective in this shape, its weights drawn from torch's generator as it stands.

        Raises:
            ValueError: If the target reads more blocks than the encoder has, a term is stated
                beside the latent target, both terms are stated, the observation is stated
                beside readings, or a latent target or a term is stated beside cells.
        """
        if self.architecture.reads_cells:
            if self.target.is_latent or self.contrast.is_stated or self.summary.is_stated:
                raise ValueError("an encoder over cells predicts their values alone")
            encoder = SetEncoder.for_vocabulary(
                self.architecture,
                self.vocabulary_size,
                dropout=dropout,
                value_embedding=nn.Linear(
                    self.architecture.input_features, self.architecture.width
                ),
            )
            return CellReconstruction(
                encoder,
                decoder_layers=self.decoder_layers,
                observation=self.observation,
                dropout=dropout,
            )
        if self.observation.is_stated:
            raise ValueError("the observation asks about cells, and the encoder reads readings")
        encoder = SetEncoder.for_vocabulary(
            self.architecture, self.vocabulary_size, dropout=dropout
        )
        if self.contrast.is_stated and self.summary.is_stated:
            raise ValueError("the contrast and the summaries are not stated together")
        if self.target.is_latent:
            if self.contrast.is_stated or self.summary.is_stated:
                raise ValueError("a term is stated beside the value target only")
            return LatentPrediction(
                encoder, predictor_layers=self.decoder_layers, target=self.target, dropout=dropout
            )
        if self.contrast.is_stated:
            return ContrastiveReconstruction(
                encoder, decoder_layers=self.decoder_layers, contrast=self.contrast, dropout=dropout
            )
        if self.summary.is_stated:
            return SummaryReconstruction(
                encoder, decoder_layers=self.decoder_layers, summary=self.summary, dropout=dropout
            )
        return MaskedReconstruction(encoder, decoder_layers=self.decoder_layers, dropout=dropout)
