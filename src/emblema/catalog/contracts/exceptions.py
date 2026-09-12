"""Malformed messages of the Catalog's published language.

They are ``ValueError`` subclasses and share nothing with the domain hierarchy: a consumer
holding a malformed reference has a bad message, not a Catalog domain failure. They do share a
base of their own, so that a consumer decoding a message can catch everything this package
raises without naming each class or widening to ``ValueError``.
"""


class CatalogContractError(ValueError):
    """A message of the Catalog's published language that cannot be read or does not hold."""


class InvalidChannelSpecError(CatalogContractError):
    pass


class InvalidCorpusVersionRefError(CatalogContractError):
    pass


class InvalidPublishedCorpusManifestError(CatalogContractError):
    pass


class MalformedManifestError(CatalogContractError):
    """Bytes that are not a manifest this codec reads, or a manifest that breaks its own rules."""
