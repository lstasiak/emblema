"""Domain exceptions of Pretraining.

Invariant violations of value objects are also ``ValueError`` so that they read naturally at the
edge.
"""


class PretrainingError(Exception):
    pass


class InvalidEncoderArchitectureError(PretrainingError, ValueError):
    pass


class InvalidMaskingStrategyError(PretrainingError, ValueError):
    pass


class InvalidLearningRateScheduleError(PretrainingError, ValueError):
    pass


class IncompatibleTalliesError(PretrainingError, ValueError):
    pass


class IncomparableRunsError(PretrainingError, ValueError):
    pass


class InvalidTrainingBudgetError(PretrainingError, ValueError):
    pass


class InvalidCheckpointPolicyError(PretrainingError, ValueError):
    pass


class InvalidExperimentConfigurationError(PretrainingError, ValueError):
    pass


class InvalidTrainingCorpusError(PretrainingError, ValueError):
    pass


class InvalidRunPositionError(PretrainingError, ValueError):
    pass


class InvalidTrainingOutcomeError(PretrainingError, ValueError):
    pass


class IncompatibleCheckpointError(PretrainingError):
    """Raised where a checkpoint is offered to a run it did not come from."""


class UnsupportedPrecisionError(PretrainingError):
    """Raised where a device cannot run the precision the experiment declared."""


class DivergedRunError(PretrainingError):
    """Raised where a run's loss or gradients stop being finite, before a step is taken on them."""


class InvalidRunSignatureError(PretrainingError, ValueError):
    pass


class InvalidPretrainingInputError(PretrainingError, ValueError):
    pass


class InvalidBackboneError(PretrainingError, ValueError):
    pass


class BackboneNotFoundError(PretrainingError):
    pass


class BackboneAlreadyDeliveredError(PretrainingError):
    """Raised where weights are delivered to a backbone that already has its artifact."""


class InvalidPretrainingOrderError(PretrainingError, ValueError):
    pass


class InvalidPretrainingResultError(PretrainingError, ValueError):
    pass


class PretrainingOrderRejectedError(PretrainingError):
    """Raised where an order cannot be placed or fulfilled: the corpus is not the one it names."""


class PretrainingResultRejectedError(PretrainingError):
    """Raised where a delivered result is not the run that was ordered, naming what differs."""


class UnreadablePublishedCorpusError(PretrainingError):
    """Raised where a published manifest or its block is not one this context can read."""


class UnreadableHandoffDocumentError(PretrainingError):
    """Raised where stored bytes are not an order or a result this context can read."""
