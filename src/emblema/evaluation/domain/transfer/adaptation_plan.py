from dataclasses import dataclass, field

from emblema.evaluation.domain.exceptions import InvalidAdaptationPlanError
from emblema.evaluation.domain.heads.head_pooling import HeadPooling
from emblema.evaluation.domain.heads.ridge_penalties import RidgePenalties
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.encoder_setting import EncoderSetting
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec
from emblema.evaluation.domain.transfer.training_regime import TrainingRegime
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class AdaptationPlan:
    """How one candidate is made out of the backbone for a task: mode, weights, schedule, seed.

    One value holds everything a cell of the curve does to the backbone, so two cells can be told
    apart by comparing plans and a run reports the plan it was made under. Which weights it
    starts from is part of the plan rather than of the mode: the control arm starts from none,
    the modes that adapt the weights from the artifact of a pretrained backbone, held by
    reference and checksum as the registry holds it, and the frozen modes from either, since
    reading an encoder at its initialisation is how a curve over pretraining starts.

    Invariants: the control arm names no weights and the modes that adapt them name some;
    low-rank updates are specified exactly when the mode adds them; penalties are named exactly
    when a head is solved in closed form, by the mode or before a trained head's first step, and
    a head so solved pools under no learnt weights, since a closed form has nothing to train
    them with; the encoder drops activations only where it runs
    inside the optimiser's loop, since elsewhere the dropout would change nothing; the encoder is
    built otherwise than its backbone (its own shape, its own value embedding) only where it starts
    from no weights, and its own shape is stated whole; a head reads below the last layer only
    where the run leaves the encoder as it is; the encoder's weights learn at less than the
    head's rate only where the mode trains them. The head reads the window's statistics per
    channel only in the network trained from nothing, with a head it trains, over the raw
    readings and every channel; where it reads them alone, nothing that only shapes the encoder
    is turned, since the encoder is never run; a head over cells reads no statistics, which
    describe the raw readings.

    Attributes:
        mode: What the backbone's weights do while the task is learnt.
        backbone: Artifact of the pretrained weights the run starts from; ``None`` for an
            encoder at its initialisation.
        schedule: How long the task is learnt, in how large a step, under what decay.
        lora: The low-rank updates, where the mode adds them; ``None`` otherwise.
        seed: Seed of everything the run draws: the head, fresh backbone weights, the low-rank
            updates, the order of windows. Apart from the seed of the draw and outside the
            schedule: another seed is a repeat of the same schedule over the same labels.
        pooling: How the states of a window become the one state the head reads; the mean
            over the window unless a variant turns it.
        ridge: The penalties a closed-form head chooses among, where one is solved; ``None``
            otherwise.
        encoder: What the encoder drops while it learns and which readings it is given; the
            standard setting unless a variant turns it.
        regime: How the run is stopped, weighted and perturbed inside its schedule; the
            standard regime unless a variant turns it.
    """

    mode: TransferMode
    backbone: ArtifactRef | None
    schedule: AdaptationSchedule
    lora: LoraSpec | None
    seed: int
    pooling: HeadPooling = field(default_factory=HeadPooling.mean)
    ridge: RidgePenalties | None = None
    encoder: EncoderSetting = field(default_factory=EncoderSetting.standard)
    regime: TrainingRegime = field(default_factory=TrainingRegime.standard)

    def __post_init__(self) -> None:
        if self.backbone is None and self.mode.needs_pretrained_weights:
            raise InvalidAdaptationPlanError(
                f"{self.mode} adapts pretrained weights and names none"
            )
        if self.backbone is not None and not self.mode.takes_pretrained_weights:
            raise InvalidAdaptationPlanError(f"{self.mode} starts from no weights and names some")
        if (self.lora is None) == self.mode.adds_low_rank_updates:
            raise InvalidAdaptationPlanError(
                f"{self.mode} "
                + (
                    "adds low-rank updates and specifies none"
                    if self.lora is None
                    else "adds no low-rank updates and specifies some"
                )
            )
        if self.regime.solves_the_head_first and self.mode.solves_the_head_in_closed_form:
            raise InvalidAdaptationPlanError(
                f"{self.mode} solves its head in closed form and takes no step to start it for"
            )
        if (self.ridge is None) == self.solves_a_head:
            raise InvalidAdaptationPlanError(
                f"{self.mode} "
                + (
                    "solves its head in closed form and names no penalties"
                    if self.ridge is None
                    else "trains its head and names penalties"
                )
            )
        if self.solves_a_head and self.pooling.pooling.learns_weights:
            raise InvalidAdaptationPlanError(
                f"{self.mode} solves its head in closed form and cannot learn a "
                f"{self.pooling.pooling} pooling"
            )
        if self.encoder.dropout > 0.0 and not self.encodes_in_the_loop:
            raise InvalidAdaptationPlanError(
                f"{self.mode} under a {self.pooling.pooling} pooling encodes every window once, "
                "so a dropout would change nothing"
            )
        if self.encoder.builds_its_own_encoder and self.backbone is not None:
            raise InvalidAdaptationPlanError(
                f"{self.mode} starts from pretrained weights, which fix its encoder's build"
            )
        if self.encoder.shape_partly_stated:
            raise InvalidAdaptationPlanError(
                "an encoder's own shape states its width, heads, layers and feed-forward width"
            )
        if not self.encoder.layer.is_last and self.mode.trains_the_encoder:
            raise InvalidAdaptationPlanError(
                f"{self.mode} trains the encoder, and reading layer {self.encoder.layer} would "
                "change which of its blocks train"
            )
        # A whole shape is judged by its own invariants as it is read.
        _ = self.encoder.shape
        if self.regime.channel_dropout > 0.0 and not self.encodes_in_the_loop:
            raise InvalidAdaptationPlanError(
                f"{self.mode} under a {self.pooling.pooling} pooling encodes every window once, "
                "so withholding channels would change nothing"
            )
        if self.regime.stop_partly_stated:
            raise InvalidAdaptationPlanError(
                "a stop states both the share it holds out and the patience it waits, and a "
                "division only with both"
            )
        if self.regime != TrainingRegime.standard() and self.mode.solves_the_head_in_closed_form:
            raise InvalidAdaptationPlanError(
                f"{self.mode} solves its head in closed form and takes no step a regime could "
                "stop, weight or perturb"
            )
        if self.regime.slows_the_encoder and not self.mode.trains_backbone_weights:
            raise InvalidAdaptationPlanError(
                f"{self.mode} trains none of the backbone's weights, so the encoder's rate "
                "would change nothing"
            )
        refusal = self.summaries_refusal(
            mode=self.mode,
            pooling=self.pooling,
            encoder=self.encoder,
            regime=self.regime,
            solves_a_head=self.solves_a_head,
        )
        if refusal is not None:
            raise InvalidAdaptationPlanError(refusal)

    @staticmethod
    def summaries_refusal(
        *,
        mode: TransferMode,
        pooling: HeadPooling,
        encoder: EncoderSetting,
        regime: TrainingRegime,
        solves_a_head: bool,
    ) -> str | None:
        """Why a head could not read the statistics per channel so, or ``None`` if it could.

        Stated once for the plan and for the catalogue that reads a variant's name, so the two
        refuse the same variants for the same reason.
        """
        if not encoder.reads_summaries:
            return None
        reading = encoder.summaries
        if mode is not TransferMode.FROM_SCRATCH:
            return (
                f"{mode} reads a backbone, and summaries {reading} take apart only the network "
                "trained from nothing"
            )
        if solves_a_head:
            return (
                f"summaries {reading} are read by a head that trains, and this one is solved over "
                "the encoder's states alone"
            )
        if encoder.grid_resolution is not None:
            return f"summaries {reading} are of the raw readings, not of readings laid on a grid"
        if encoder.reads_cells:
            return f"summaries {reading} are of the raw readings, not of cells"
        if regime.channel_dropout > 0.0:
            return f"summaries {reading} would still show the channels a batch withholds"
        if encoder.reads_tokens:
            return None
        shaping = {
            "a dropout": encoder.dropout > 0.0,
            "an encoder of its own build": encoder.builds_its_own_encoder,
            "a clip": encoder.value_clip is not None,
            "a layer below the last": not encoder.layer.is_last,
            "a pooling of its own": pooling != HeadPooling.mean(),
            "a slower encoder": regime.slows_the_encoder,
        }
        turned = [label for label, holds in shaping.items() if holds]
        if not turned:
            return None
        return (
            f"summaries {reading} never run the encoder, so {', '.join(turned)} would change "
            "nothing"
        )

    @property
    def solves_a_head(self) -> bool:
        """Whether a head is solved in closed form: as the mode's answer or as a run's start."""
        return self.mode.solves_the_head_in_closed_form or self.regime.solves_the_head_first

    @property
    def encodes_in_the_loop(self) -> bool:
        """Whether the encoder runs inside the optimiser's loop, rather than once per window."""
        return self.mode.encodes_in_the_loop(self.pooling)

    def parameters(self) -> dict[str, str | int | float]:
        """The plan flattened to scalars, in a fixed order, for whoever reports a run.

        Absent parts are rendered as empty text and zeros rather than left out, so every plan
        renders the same columns and two plans can be compared by comparing this mapping. The
        seed is named ``run_seed``, so a row that also carries the seed of the draw reads
        unambiguously.
        """
        return {
            "mode": str(self.mode),
            "backbone": "" if self.backbone is None else self.backbone.key,
            "run_seed": self.seed,
            "epochs": self.schedule.epochs,
            "min_steps": self.schedule.min_steps,
            "batch_size": self.schedule.batch_size,
            "learning_rate": float(self.schedule.learning_rate),
            "weight_decay": float(self.schedule.weight_decay),
            "warmup_fraction": float(self.schedule.warmup_fraction),
            "final_lr_fraction": float(self.schedule.final_lr_fraction),
            **self.pooling.parameters(),
            "statics": str(self.pooling.statics),
            "queries": self.pooling.queries,
            "lora_rank": 0 if self.lora is None else self.lora.rank,
            "lora_alpha": 0.0 if self.lora is None else float(self.lora.alpha),
            "lora_dropout": 0.0 if self.lora is None else float(self.lora.dropout),
            "lora_targets": "" if self.lora is None else " ".join(self.lora.targets),
            "ridge_penalties": "" if self.ridge is None else str(self.ridge),
            **self.encoder.parameters(),
            **self.regime.parameters(),
        }
