"""Failures a shared port reports to its caller, whichever adapter stands behind it.

Adapters translate their technology's errors into these; a use case never sees a driver
exception.
"""


class ArtifactStoreError(Exception):
    """Base of the failures an ``ArtifactStore`` reports."""


class ArtifactNotFoundError(ArtifactStoreError):
    """Nothing is stored under the key of the given reference."""


class ArtifactIntegrityError(ArtifactStoreError):
    """The stored bytes do not hash to the checksum of the given reference."""


class JobQueueError(Exception):
    """A job could not be handed to the queue; nothing was accepted."""
