"""Malformed messages of the Catalog's published language.

They are ``ValueError`` subclasses and share nothing with the domain hierarchy: a consumer
holding a malformed reference has a bad message, not a Catalog domain failure.
"""


class InvalidChannelSpecError(ValueError):
    pass


class InvalidCorpusVersionRefError(ValueError):
    pass
