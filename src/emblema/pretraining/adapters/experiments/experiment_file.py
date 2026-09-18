"""The file an experiment is stated in, and the sections it is made of.

One class per section rather than one flat table, because the sections are what the run is made
of: what hides, how long, how often to checkpoint. They are read together and never apart, so
they live together here.
"""

import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from emblema.config.compute_tiers import ComputeTiers
from emblema.pretraining.adapters.encoder.tier_architecture import architecture_of
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.pretraining.domain.training.checkpoint_policy import CheckpointPolicy
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.objective_loss import LossKind, ObjectiveLoss
from emblema.pretraining.domain.training.precision import Precision
from emblema.pretraining.domain.training.training_budget import TrainingBudget
from emblema.shared.kernel.compute import ComputeTier


class _Section(BaseModel):
    """What every section of the file is: fixed once read, and closed to keys nobody reads."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class _Shape(_Section):
    """What the run overrides of the shape its tier states, field by field.

    A machine slower than the tier assumes runs a smaller model, and that is a legitimate result
    as long as the shape it ran is written down rather than passed on a command line.
    """

    width: int | None = None
    heads: int | None = None
    layers: int | None = None
    feedforward_width: int | None = None
    time_frequencies: int | None = None

    def over(self, tier: EncoderArchitecture) -> EncoderArchitecture:
        return EncoderArchitecture(
            width=tier.width if self.width is None else self.width,
            heads=tier.heads if self.heads is None else self.heads,
            layers=tier.layers if self.layers is None else self.layers,
            feedforward_width=(
                tier.feedforward_width if self.feedforward_width is None else self.feedforward_width
            ),
            time_frequencies=(
                tier.time_frequencies if self.time_frequencies is None else self.time_frequencies
            ),
        )


class _Masking(_Section):
    channel_rate: float
    block_rate: float
    block_span: float
    token_rate: float

    def strategy(self) -> MaskingStrategy:
        return MaskingStrategy(
            channel_rate=self.channel_rate,
            block_rate=self.block_rate,
            block_span=self.block_span,
            token_rate=self.token_rate,
        )


class _Objective(_Section):
    """What the objective counts a miss as; a file that says nothing is scored by the square."""

    kind: LossKind = LossKind.MSE
    huber_delta: float = 0.0

    def loss(self) -> ObjectiveLoss:
        return ObjectiveLoss(kind=self.kind, huber_delta=self.huber_delta)


class _Budget(_Section):
    epochs: int
    batch_size: int
    accumulation_steps: int
    learning_rate: float
    warmup_epochs: float
    final_lr_fraction: float
    seed: int

    def budget(self) -> TrainingBudget:
        return TrainingBudget(
            epochs=self.epochs,
            batch_size=self.batch_size,
            accumulation_steps=self.accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_epochs=self.warmup_epochs,
            final_lr_fraction=self.final_lr_fraction,
            seed=self.seed,
        )


class _Checkpoint(_Section):
    every_steps: int

    def policy(self) -> CheckpointPolicy:
        return CheckpointPolicy(every_steps=self.every_steps)


class ExperimentFile(_Section):
    """An experiment as it is written down: every parameter of scale, in one file, under a tier.

    The file is the source of a run. The shape and the share of the corpus come from the tier
    unless the file says otherwise, and everything else is stated outright, so that reproducing a
    run means running the same file rather than remembering which flags were passed. What the
    values are allowed to be is the domain's business: this reads the file, refuses keys nobody
    reads, and lets the value objects say what is out of range.

    Attributes:
        name: What the experiment is called; runs of it are grouped under this name.
        tier: Hardware class the run declares, and the shape it takes unless overridden.
        corpora: The published corpora the run reads, in the order their vocabulary was chained;
            one for a run over one corpus.
        corpus_fraction: Share of each corpus's training units the run reads; the tier's unless
            stated.
        precision: What the forward and backward pass are computed at.
        dropout: Dropout of the encoder and the decoder.
        decoder_layers: Blocks of the decoder thrown away when the run ends.
        shape: What the run overrides of the tier's shape, where it does.
        masking: What the objective hides.
        objective: What it counts a miss as; the squared error unless the file says otherwise,
            which is what every run before the reading existed was scored by.
        budget: How long the run trains and in how large a step.
        checkpoint: How often resumable state is written.
    """

    name: str
    tier: ComputeTier
    corpora: tuple[str, ...] = Field(min_length=1)
    corpus_fraction: float | None = None
    precision: Precision
    dropout: float
    decoder_layers: int
    shape: _Shape | None = None
    masking: _Masking
    objective: _Objective = _Objective()
    budget: _Budget
    checkpoint: _Checkpoint

    @classmethod
    def load(cls, path: Path) -> Self:
        """The experiment stated in ``path``.

        Raises:
            FileNotFoundError: If there is no such file.
            tomllib.TOMLDecodeError: If it is not TOML.
            pydantic.ValidationError: If it states something other than an experiment.
        """
        with path.open("rb") as handle:
            return cls.model_validate(tomllib.load(handle))

    def configuration(self, tiers: ComputeTiers | None = None) -> ExperimentConfiguration:
        """The configuration this file states, with the shape its tier gives it.

        Raises:
            InvalidExperimentConfigurationError: If what the file states is not a configuration.
            InvalidCorpusShareError: If the share of the corpus is not one a run could read.
            InvalidEncoderArchitectureError: If the shape it states is not an architecture.
            InvalidMaskingStrategyError: If the strategy hides everything or nothing.
            InvalidObjectiveLossError: If the reading states a knee it does not read, or none.
            InvalidTrainingBudgetError: If the budget is not one a run could follow.
            InvalidCheckpointPolicyError: If the interval is not positive.
        """
        profile = (ComputeTiers.load() if tiers is None else tiers).profile(self.tier)
        architecture = architecture_of(profile)
        return ExperimentConfiguration(
            name=self.name,
            tier=self.tier,
            corpus_fraction=(
                profile.corpus_fraction if self.corpus_fraction is None else self.corpus_fraction
            ),
            architecture=architecture if self.shape is None else self.shape.over(architecture),
            dropout=self.dropout,
            decoder_layers=self.decoder_layers,
            masking=self.masking.strategy(),
            loss=self.objective.loss(),
            budget=self.budget.budget(),
            precision=self.precision,
            checkpoint=self.checkpoint.policy(),
        )
