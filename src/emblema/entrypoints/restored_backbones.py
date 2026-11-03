from collections.abc import Sequence
from dataclasses import replace

from torch import nn

from emblema.evaluation.adapters.detection.masked_cells import MaskedCells
from emblema.evaluation.domain.detection.hidden_cells_choice import HiddenCellsChoice
from emblema.evaluation.domain.exceptions import UnknownBackboneError, UnreadablePretextError
from emblema.evaluation.domain.transfer.encoder_setting import TimeEncoding, ValueEmbedding
from emblema.evaluation.domain.transfer.encoder_shape import EncoderShape
from emblema.pretraining.adapters.encoder.learned_channel_embedding import (
    LearnedChannelEmbedding,
)
from emblema.pretraining.adapters.encoder.learned_time_encoding import LearnedTimeEncoding
from emblema.pretraining.adapters.encoder.nonlinear_value_embedding import (
    NonlinearValueEmbedding,
)
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.pretext_reading import PretextReading
from emblema.pretraining.adapters.training.exceptions import (
    UnanswerablePretextError,
    UnreadableTrainedModelError,
)
from emblema.pretraining.adapters.training.trained_model import TrainedModel
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import N_FEATURES, TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore


class RestoredBackbones:
    """Evaluation's encoders out of Pretraining's stored model: the seam the process wires.

    Evaluation asks for an encoder through a shape of call and may not import the module that
    answers it; a process knows both sides, so this is where the trained model Pretraining stored
    becomes the backbone Evaluation adapts. One instance stands for one stored model: the
    pretrained encoder is that model's, a fresh one is the same shape over the same vocabulary
    with weights drawn anew, and a request for other weights is a wiring error rather than a
    second download. The same model answers a detector that asks what its pretext would fill
    in, its decoder included, under the masks it was trained with.
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
        self._readings: dict[str, PretextReading] = {}

    @property
    def width(self) -> int:
        return self._trained.architecture.width

    @property
    def layers(self) -> int:
        return self._trained.architecture.layers

    @property
    def bins(self) -> int:
        return self._trained.architecture.bins

    @property
    def huber_delta(self) -> float:
        loss = self._trained.loss
        return 0.0 if loss is None else loss.huber_delta

    def masked(
        self,
        weights: ArtifactRef,
        windows: Sequence[TokenWindow],
        seeds: Sequence[int],
        *,
        answered: bool,
        device: str,
        hidden: HiddenCellsChoice = HiddenCellsChoice.OWN,
    ) -> list[MaskedCells]:
        """The stored objective's draws over the windows, or their tails, read back once per device.

        Raises:
            UnknownBackboneError: If other weights are asked for than this was built over.
            UnreadablePretextError: If the stored model cannot say what it hid, its weights do not
                fit their shape, or a window holds a channel it learnt no row for.
            ValueError: If the windows and seeds are not paired, or a seed lies outside
                ``[0, 2**63)``: a caller's error, not the backbone's.
        """
        self._require(weights)
        try:
            reading = self._readings.get(device)
            if reading is None:
                reading = self._readings[device] = PretextReading.of(self._trained, device=device)
            drawn = reading.drawn(
                windows, seeds, answered=answered, tail=hidden is HiddenCellsChoice.TAIL
            )
        except (UnanswerablePretextError, UnreadableTrainedModelError) as error:
            raise UnreadablePretextError(str(error)) from error
        return [
            MaskedCells(
                full=read.full,
                visible=read.visible,
                hidden=read.hidden,
                channel_ids=read.channel_ids,
                timeless=read.timeless,
                answers=read.answers,
            )
            for read in drawn
        ]

    def _require(self, weights: ArtifactRef) -> None:
        if weights != self._weights:
            raise UnknownBackboneError(
                f"this process serves the backbone {self._weights.key}, not {weights.key}"
            )

    def pretrained(
        self, weights: ArtifactRef, *, vocabulary_size: int, dropout: float
    ) -> nn.Module:
        """The stored encoder in evaluation mode, grown to the task's vocabulary where it is short.

        Raises:
            UnknownBackboneError: If other weights are asked for than this was built over.
        """
        self._require(weights)
        return self._trained.build(dropout=dropout).encoder.grown_to(vocabulary_size).eval()

    @staticmethod
    def drawn(
        architecture: EncoderArchitecture,
        vocabulary_size: int,
        *,
        dropout: float,
        value_embedding: ValueEmbedding,
        time_encoding: TimeEncoding,
        input_features: int,
    ) -> SetEncoder:
        """An encoder of ``architecture`` with weights drawn anew, its input modules as asked.

        The standard composition where the value is linear over the raw tokens' features and the
        time fixed, so an encoder asked for as before is drawn exactly as before; otherwise the
        modules Evaluation's knobs name take their places behind the same call.
        """
        width = architecture.width
        value: nn.Module | None
        if value_embedding is ValueEmbedding.NONLINEAR:
            value = NonlinearValueEmbedding(width, inputs=input_features)
        elif input_features != N_FEATURES:
            value = nn.Linear(input_features, width)
        else:
            value = None
        if time_encoding is TimeEncoding.FOURIER:
            return SetEncoder.for_vocabulary(
                architecture, vocabulary_size, dropout=dropout, value_embedding=value
            )
        return SetEncoder(
            architecture,
            channel_embedding=LearnedChannelEmbedding(vocabulary_size, width),
            time_encoding=LearnedTimeEncoding(architecture.time_frequencies, width),
            dropout=dropout,
            value_embedding=value,
        )

    def fresh(
        self,
        *,
        vocabulary_size: int,
        dropout: float,
        value_embedding: ValueEmbedding = ValueEmbedding.LINEAR,
        shape: EncoderShape | None = None,
        time_encoding: TimeEncoding = TimeEncoding.FOURIER,
        input_features: int = N_FEATURES,
        bins: int = 0,
    ) -> nn.Module:
        # The input part is the run's layout's, not the stored model's: a fresh encoder over
        # the raw readings beside a backbone over cells reads two features, and says so.
        architecture = replace(self._trained.architecture, bins=bins)
        if shape is not None:
            architecture = replace(
                architecture,
                width=shape.width,
                heads=shape.heads,
                layers=shape.layers,
                feedforward_width=shape.feedforward_width,
            )
        return self.drawn(
            architecture,
            max(vocabulary_size, self._trained.vocabulary_size),
            dropout=dropout,
            value_embedding=value_embedding,
            time_encoding=time_encoding,
            input_features=input_features,
        )
