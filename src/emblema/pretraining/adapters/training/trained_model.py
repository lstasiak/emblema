import io
from dataclasses import dataclass, field
from typing import Any, Self

import torch

from emblema.pretraining.adapters.objective.objective_shape import ObjectiveShape
from emblema.pretraining.adapters.objective.pretraining_objective import PretrainingObjective
from emblema.pretraining.adapters.training.exceptions import (
    UNREADABLE_BYTES,
    UnreadableTrainedModelError,
)
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.exceptions import IncompatibleStartError
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.objective_target import ObjectiveTarget, TargetKind

# The one tensor a continued run carries over by rows rather than whole: the channel table,
# whose height is the vocabulary the run reads under, not the shape.
CHANNEL_TABLE = "channel_embedding.table.weight"


@dataclass(frozen=True)
class TrainedModel:
    """What a finished run keeps: the backbone, the objective that trained it, and their shape.

    The optimiser's state is left out — it belongs to a run that is over — and the rest of the
    objective is kept even though it is thrown away for transfer, because without it the run's
    own predictions cannot be reproduced, and a diagnostic that scores something other than what
    was stored is scoring the wrong thing. The shape travels with the weights so the artifact
    rebuilds a model on its own, without the configuration that produced it; so does the target,
    which says which objective the weights are of. Bytes written before a target could be stated
    are of the objective every run then had, the one predicting values.

    Attributes:
        architecture: Shape of the encoder the weights load into.
        vocabulary_size: Channels of the corpus the run read, which sizes the channel table.
        decoder_layers: Blocks of the decoder, or of the predictor.
        weights: State of the whole objective, on the host.
        target: What the objective predicted, which picks the objective the weights load into.
    """

    architecture: EncoderArchitecture
    vocabulary_size: int
    decoder_layers: int
    weights: dict[str, Any]
    target: ObjectiveTarget = field(default_factory=ObjectiveTarget.value)

    @classmethod
    def of(
        cls,
        configuration: ExperimentConfiguration,
        vocabulary_size: int,
        model: PretrainingObjective,
    ) -> Self:
        """The model as it ended, with its weights copied to the host."""
        return cls(
            architecture=configuration.architecture,
            vocabulary_size=vocabulary_size,
            decoder_layers=configuration.decoder_layers,
            weights={key: value.detach().to("cpu") for key, value in model.state_dict().items()},
            target=configuration.target,
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
                "target": str(self.target.kind),
                "target_layers": self.target.layers,
                "momentum": self.target.momentum,
                "final_momentum": self.target.final_momentum,
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
                target=ObjectiveTarget(
                    kind=TargetKind(stored.get("target", "value")),
                    layers=stored.get("target_layers", 0),
                    momentum=stored.get("momentum", 0.0),
                    final_momentum=stored.get("final_momentum", 0.0),
                ),
            )
        except UNREADABLE_BYTES as error:
            raise UnreadableTrainedModelError(f"not a trained model this can read: {error}") from (
                error
            )

    def build(self, *, dropout: float = 0.0) -> PretrainingObjective:
        """The objective rebuilt on the host, in evaluation mode, holding these weights.

        Dropout is zero unless asked for: a model being read back is being asked what it
        predicts, and the dropout it trained under is a property of the run, not of the weights.
        A process that goes on training the weights names the dropout of its own run.

        Raises:
            UnreadableTrainedModelError: If the weights do not fit the shape stored with them.
        """
        shape = ObjectiveShape(
            architecture=self.architecture,
            vocabulary_size=self.vocabulary_size,
            decoder_layers=self.decoder_layers,
            target=self.target,
        )
        try:
            model = shape.build(dropout=dropout)
            model.load_state_dict(self.weights)
        except (RuntimeError, ValueError) as error:
            raise UnreadableTrainedModelError(
                f"weights do not fit the shape stored: {error}"
            ) from (error)
        return model.eval()

    def require_shape_of(self, configuration: ExperimentConfiguration) -> None:
        """Refuse a run of ``configuration`` that would continue these weights in another shape.

        Raises:
            IncompatibleStartError: If the shape, the decoder or the target stored with the
                weights is not the run's.
        """
        for label, stored, asked in (
            ("shape", self.architecture, configuration.architecture),
            ("decoder", self.decoder_layers, configuration.decoder_layers),
            ("target", self.target, configuration.target),
        ):
            if stored != asked:
                raise IncompatibleStartError(
                    f"the stored model has another {label} than {configuration.name!r} states: "
                    f"{stored} against {asked}"
                )

    def carried_into(self, model: PretrainingObjective) -> None:
        """Put these weights into ``model``, an objective of this shape over any vocabulary.

        Every tensor is copied whole but the channel table, which is copied row by row as far as
        both tables reach: a row past these weights keeps what the model drew, a row past the
        model's table is left behind. A run that continues a backbone over a corpus the backbone
        never read is where the first happens; one over a corpus from the middle of the
        backbone's chain is where the second does.

        Raises:
            IncompatibleStartError: If a tensor of the model is missing from these weights or
                differs in shape, which is a model of another objective or architecture.
        """
        carried = {}
        for key, current in model.state_dict().items():
            stored = self.weights.get(key)
            if stored is None:
                raise IncompatibleStartError(f"the stored model has no tensor {key!r}")
            if stored.shape == current.shape:
                carried[key] = stored
            elif key.endswith(CHANNEL_TABLE) and stored.shape[1:] == current.shape[1:]:
                rows = min(stored.shape[0], current.shape[0])
                table = current.detach().clone()
                table[:rows] = stored[:rows]
                carried[key] = table
            else:
                raise IncompatibleStartError(
                    f"the stored tensor {key!r} is of shape {tuple(stored.shape)}, the run's of "
                    f"{tuple(current.shape)}"
                )
        unknown = sorted(set(self.weights) - set(carried))
        if unknown:
            raise IncompatibleStartError(f"the stored model has tensors the run has not: {unknown}")
        model.load_state_dict(carried)
