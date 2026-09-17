from collections.abc import Mapping

from emblema.pretraining.adapters.documents.fields import Document, Fields
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.pretraining.domain.training.checkpoint_policy import CheckpointPolicy
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.objective_loss import LossKind, ObjectiveLoss
from emblema.pretraining.domain.training.precision import Precision
from emblema.pretraining.domain.training.training_budget import TrainingBudget
from emblema.shared.kernel.compute import ComputeTier


class ExperimentConfigurationDocument:
    """A configuration as a JSON object, nested as the value objects are.

    The resolved configuration rather than the experiment file: the shape is stated outright,
    not left to the tier, so a document rebuilds the run without the tier table of the code that
    wrote it. The same object is stored beside a backbone and carried inside an order, so a run
    reads the same way wherever it is looked up.
    """

    def encode(self, configuration: ExperimentConfiguration) -> Document:
        architecture, masking, budget = (
            configuration.architecture,
            configuration.masking,
            configuration.budget,
        )
        return {
            "name": configuration.name,
            "tier": str(configuration.tier),
            "corpus_fraction": configuration.corpus_fraction,
            "architecture": {
                "width": architecture.width,
                "heads": architecture.heads,
                "layers": architecture.layers,
                "feedforward_width": architecture.feedforward_width,
                "time_frequencies": architecture.time_frequencies,
            },
            "dropout": configuration.dropout,
            "decoder_layers": configuration.decoder_layers,
            "masking": {
                "channel_rate": masking.channel_rate,
                "block_rate": masking.block_rate,
                "block_span": masking.block_span,
                "token_rate": masking.token_rate,
            },
            "objective": {
                "kind": str(configuration.loss.kind),
                "huber_delta": configuration.loss.huber_delta,
            },
            "budget": {
                "epochs": budget.epochs,
                "batch_size": budget.batch_size,
                "accumulation_steps": budget.accumulation_steps,
                "learning_rate": budget.learning_rate,
                "warmup_epochs": budget.warmup_epochs,
                "final_lr_fraction": budget.final_lr_fraction,
                "seed": budget.seed,
            },
            "precision": str(configuration.precision),
            "checkpoint": {"every_steps": configuration.checkpoint.every_steps},
        }

    def decode(self, document: Mapping[str, object]) -> ExperimentConfiguration:
        """The configuration the object states.

        Raises:
            ValueError: If a field is missing or mistyped, or what is stated is not a
                configuration; the domain's refusals are ``ValueError`` too.
        """
        fields = Fields(document)
        architecture, masking, budget = (
            fields.fields("architecture"),
            fields.fields("masking"),
            fields.fields("budget"),
        )
        objective = fields.fields("objective")
        return ExperimentConfiguration(
            name=fields.text("name"),
            tier=ComputeTier(fields.text("tier")),
            corpus_fraction=fields.number("corpus_fraction"),
            architecture=EncoderArchitecture(
                width=architecture.integer("width"),
                heads=architecture.integer("heads"),
                layers=architecture.integer("layers"),
                feedforward_width=architecture.integer("feedforward_width"),
                time_frequencies=architecture.integer("time_frequencies"),
            ),
            dropout=fields.number("dropout"),
            decoder_layers=fields.integer("decoder_layers"),
            masking=MaskingStrategy(
                channel_rate=masking.number("channel_rate"),
                block_rate=masking.number("block_rate"),
                block_span=masking.number("block_span"),
                token_rate=masking.number("token_rate"),
            ),
            loss=ObjectiveLoss(
                kind=LossKind(objective.text("kind")),
                huber_delta=objective.number("huber_delta"),
            ),
            budget=TrainingBudget(
                epochs=budget.integer("epochs"),
                batch_size=budget.integer("batch_size"),
                accumulation_steps=budget.integer("accumulation_steps"),
                learning_rate=budget.number("learning_rate"),
                warmup_epochs=budget.number("warmup_epochs"),
                final_lr_fraction=budget.number("final_lr_fraction"),
                seed=budget.integer("seed"),
            ),
            precision=Precision(fields.text("precision")),
            checkpoint=CheckpointPolicy(
                every_steps=fields.fields("checkpoint").integer("every_steps")
            ),
        )
