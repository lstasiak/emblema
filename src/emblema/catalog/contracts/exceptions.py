"""Malformed messages of the Catalog's published language.

They are ``ValueError`` subclasses and share nothing with the domain hierarchy: a consumer
holding a malformed reference has a bad message, not a Catalog domain failure.
"""


class InvalidChannelSpecError(ValueError):
    pass


class InvalidCorpusVersionRefError(ValueError):
    pass


class InvalidPublishedCorpusManifestError(ValueError):
    pass


class MalformedManifestError(ValueError):
    """Bytes that are not a manifest this codec reads, or a manifest that breaks its own rules."""
