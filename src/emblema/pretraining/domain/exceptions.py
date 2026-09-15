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
