"""Malformed messages of the Evaluation context's published language.

They are ``ValueError`` subclasses and share nothing with the domain hierarchy: a consumer
holding a malformed reference has a bad message, not an Evaluation domain failure. They share a
base of their own, so that a consumer reading a message can catch everything this package raises
without naming each class or widening to ``ValueError``.
"""


class EvaluationContractError(ValueError):
    """A message of the Evaluation context's published language that does not hold."""


class InvalidCandidateRefError(EvaluationContractError):
    pass


class InvalidCandidateMetricError(EvaluationContractError):
    pass


class InvalidKeptRepresentationError(EvaluationContractError):
    pass


class InvalidKeptCandidateManifestError(EvaluationContractError):
    pass


class MalformedKeptCandidateManifestError(EvaluationContractError):
    pass
