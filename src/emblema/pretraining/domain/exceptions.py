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
