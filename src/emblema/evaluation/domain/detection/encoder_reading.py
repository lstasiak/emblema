from dataclasses import dataclass, field

from emblema.evaluation.domain.exceptions import InvalidDetectorRecipeError
from emblema.evaluation.domain.heads.head_pooling import HeadPooling
from emblema.evaluation.domain.transfer.encoder_setting import EncoderSetting
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class EncoderReading:
    """Which frozen encoder a detector reads a window through, and how its states are pooled.

    A detector reads an encoder as a probe does, once per window and with nothing trained: the
    encoder is the pretrained one or the same shape at its initialisation, read at the layer the
    setting names, and the states per token are pooled by a scheme with no weights of its own,
    since nothing here learns.

    Invariants: the pooling learns no weights; the encoder drops nothing, since it never learns;
    the readings are the tokens as the encoder was trained on them, neither gridded, clipped
    nor replaced by the summaries, and a pretrained encoder keeps its own build.

    Attributes:
        architecture: Artifact of the model whose shape the encoder has, whether it starts from
            its weights or draws them anew.
        backbone: Artifact of the pretrained weights; ``None`` for the shape at its
            initialisation.
        encoder: Which readings the encoder is given and which layer it is read at.
        pooling: How the states of a window become the one state the detector reads.
    """

    architecture: ArtifactRef
    backbone: ArtifactRef | None
    encoder: EncoderSetting = field(default_factory=EncoderSetting.standard)
    pooling: HeadPooling = field(default_factory=HeadPooling.mean)

    def __post_init__(self) -> None:
        if self.pooling.pooling.learns_weights:
            raise InvalidDetectorRecipeError(
                f"a detector learns nothing, so it cannot pool under {self.pooling.pooling}"
            )
        if self.encoder.dropout > 0.0:
            raise InvalidDetectorRecipeError("a detector learns nothing, so it drops nothing")
        if self.encoder.grid_resolution is not None or self.encoder.value_clip is not None:
            raise InvalidDetectorRecipeError(
                "a detector reads the tokens as they are, neither gridded nor clipped"
            )
        if self.encoder.reads_summaries:
            raise InvalidDetectorRecipeError(
                "a detector over an encoder's states reads no summaries; the summaries are a "
                "detector of their own"
            )
        if self.backbone is not None and self.encoder.builds_its_own_encoder:
            raise InvalidDetectorRecipeError(
                "pretrained weights fix the encoder's build; only its layer and pooling turn"
            )
