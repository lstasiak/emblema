from dataclasses import dataclass
from typing import Self

from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.exceptions import InvalidExperimentConfigurationError
from emblema.pretraining.domain.learning_rate_schedule import LearningRateSchedule
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.pretraining.domain.training.checkpoint_policy import CheckpointPolicy
from emblema.pretraining.domain.training.corpus_share import CorpusShare
from emblema.pretraining.domain.training.precision import Precision
from emblema.pretraining.domain.training.training_budget import TrainingBudget
from emblema.shared.kernel.compute import ComputeTier


@dataclass(frozen=True, kw_only=True)
class ExperimentConfiguration:
    """Everything that decides what a run does, stated once and reported with its result.

    One value holds the shape of the model, what the objective hides, how long the run trains and
    how precisely, so that no parameter of scale is left in code and two runs can be told apart by
    comparing configurations rather than diffs. The tier it declares is what a figure is signed
    with: a smaller model over a tenth of the corpus is a legitimate result at tier S and a
    misleading one unsigned.

    Invariants: the name is non-empty and carries no surrounding whitespace; dropout lies in
    ``[0, 1)``, since a run that drops everything learns nothing; the decoder has at least one
    layer; the corpus fraction is a share a run could read.

    Attributes:
        name: What the experiment is called, and what its runs are grouped under.
        tier: Hardware class the run declares, which the reported result is signed with.
        corpus_fraction: Share of the corpus's training units the run reads; the validation side
            is read whole whatever it is. A parameter of scale like the shape, so a run over a
            tenth of the corpus says so wherever the configuration is reported.
        architecture: Shape of the encoder being pretrained.
        dropout: Dropout of the encoder and the decoder; a run parameter, not part of the shape.
        decoder_layers: Blocks of the decoder that is thrown away when the run ends.
        masking: What the objective hides, and in what shapes.
        budget: How long the run trains and in how large a step.
        precision: What the forward and backward pass are computed at.
        checkpoint: How often resumable state is written.
    """

    name: str
    tier: ComputeTier
    corpus_fraction: float
    architecture: EncoderArchitecture
    dropout: float
    decoder_layers: int
    masking: MaskingStrategy
    budget: TrainingBudget
    precision: Precision
    checkpoint: CheckpointPolicy

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise InvalidExperimentConfigurationError(
                "name must be non-empty without surrounding whitespace"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise InvalidExperimentConfigurationError(
                f"dropout must lie in [0, 1), got {self.dropout}"
            )
        if self.decoder_layers < 1:
            raise InvalidExperimentConfigurationError(
                f"decoder_layers must be positive, got {self.decoder_layers}"
            )
        # Building the share runs its invariant: a fraction no run could read is no configuration.
        _ = self.corpus_share

    @property
    def corpus_share(self) -> CorpusShare:
        """Which part of the corpus a run reads: the fraction, ranked by the run's seed.

        Raises:
            InvalidCorpusShareError: If the fraction does not lie in ``(0, 1]``.
        """
        return CorpusShare(fraction=self.corpus_fraction, seed=self.budget.seed)

    def schedule(self, batches: int) -> LearningRateSchedule:
        """The learning rate over the run, for an epoch of ``batches`` micro-batches."""
        return self.budget.schedule(batches)

    def parameters(self) -> dict[str, str | int | float]:
        """The configuration flattened to scalars, in a fixed order.

        One rendering serves everyone who has to state a run rather than run it: the tracker logs
        these as the run's parameters, a run's signature digests them, and a checkpoint is refused
        for a run whose parameters differ. Nested values are prefixed by what they belong to, so
        two configurations differing anywhere differ in this mapping. Rates are rendered as floats
        whatever they were built as: an integer zero and a float zero are one configuration and
        must digest to one signature.
        """
        return {
            "name": self.name,
            "tier": str(self.tier),
            "corpus_fraction": float(self.corpus_fraction),
            "width": self.architecture.width,
            "heads": self.architecture.heads,
            "layers": self.architecture.layers,
            "feedforward_width": self.architecture.feedforward_width,
            "time_frequencies": self.architecture.time_frequencies,
            "dropout": float(self.dropout),
            "decoder_layers": self.decoder_layers,
            "channel_rate": float(self.masking.channel_rate),
            "block_rate": float(self.masking.block_rate),
            "block_span": float(self.masking.block_span),
            "token_rate": float(self.masking.token_rate),
            "expected_hidden_ratio": float(self.masking.expected_ratio),
            "epochs": self.budget.epochs,
            "batch_size": self.budget.batch_size,
            "accumulation_steps": self.budget.accumulation_steps,
            "effective_batch_size": self.budget.effective_batch_size,
            "learning_rate": float(self.budget.learning_rate),
            "warmup_epochs": float(self.budget.warmup_epochs),
            "final_lr_fraction": float(self.budget.final_lr_fraction),
            "seed": self.budget.seed,
            "precision": str(self.precision),
            "checkpoint_every_steps": self.checkpoint.every_steps,
        }

    def differences_from(self, other: Self) -> tuple[str, ...]:
        """Names of the parameters on which this configuration differs from ``other``."""
        stated, expected = self.parameters(), other.parameters()
        return tuple(key for key in expected if stated[key] != expected[key])
