"""Invariant violations of the shared kernel value objects.

They are ``ValueError`` subclasses: the kernel belongs to no bounded context, so it has no
context-level error hierarchy to hang them on.
"""


class InvalidEntityIdError(ValueError):
    pass


class InvalidChecksumError(ValueError):
    pass


class InvalidArtifactRefError(ValueError):
    pass


class InvalidUtcDateTimeError(ValueError):
    pass
