"""Domain exceptions of the Data Catalog.

Invariant violations of value objects and constructors are also ``ValueError`` so that they
read naturally at the edge; lifecycle violations of the aggregate and failures reported by the
context's ports are plain domain errors.
"""


class CatalogError(Exception):
    pass


class InvalidChannelSchemaError(CatalogError, ValueError):
    pass


class InvalidLicenceError(CatalogError, ValueError):
    pass


class InvalidCorpusSourceError(CatalogError, ValueError):
    pass


class InvalidCorpusContentError(CatalogError, ValueError):
    pass


class InvalidCorpusError(CatalogError, ValueError):
    pass


class InvalidCorpusVersionError(CatalogError, ValueError):
    pass


class CorpusVersionNotFoundError(CatalogError):
    pass


class CorpusVersionAlreadyExistsError(CatalogError):
    pass


class CorpusVersionFrozenError(CatalogError):
    pass


class CorpusVersionNotValidatedError(CatalogError):
    pass


class CorpusVersionNotFrozenError(CatalogError):
    pass


class SameDataAlreadyFrozenError(CatalogError):
    pass


class CorpusNotFoundError(CatalogError):
    pass


class CorpusNameTakenError(CatalogError):
    pass


class CorpusReadError(CatalogError):
    pass


class CorpusDataNotFoundError(CorpusReadError):
    pass


class MalformedCorpusDataError(CorpusReadError):
    pass
