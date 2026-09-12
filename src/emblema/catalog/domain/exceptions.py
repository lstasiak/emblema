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


class InvalidUnitKeyError(CatalogError, ValueError):
    pass


class InvalidObservationError(CatalogError, ValueError):
    pass


class InvalidStaticFeatureError(CatalogError, ValueError):
    pass


class InvalidTimeExtentError(CatalogError, ValueError):
    pass


class InvalidCorpusUnitError(CatalogError, ValueError):
    pass


class InvalidWindowSpecError(CatalogError, ValueError):
    pass


class InvalidChannelVocabularyError(CatalogError, ValueError):
    pass


class InvalidChannelStatisticsError(CatalogError, ValueError):
    pass


class InvalidTokenisationSchemeError(CatalogError, ValueError):
    pass


class InvalidUnitSplitError(CatalogError, ValueError):
    pass


class InvalidTokenisationManifestError(CatalogError, ValueError):
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


class CorpusDataChangedError(CatalogError):
    pass


class CorpusNotFoundError(CatalogError):
    pass


class CorpusNameTakenError(CatalogError):
    pass


class UnknownChannelError(CatalogError):
    pass


class ChannelRedeclaredError(CatalogError):
    pass


class ChannelAlreadyFittedError(CatalogError):
    pass


class MissingChannelStatisticsError(CatalogError):
    pass


class CorpusReadError(CatalogError):
    pass


class CorpusDataNotFoundError(CorpusReadError):
    pass


class MalformedCorpusDataError(CorpusReadError):
    pass


class UnknownUnitError(CorpusReadError):
    pass


class TokenisationError(CatalogError):
    pass


class ObservationOutOfOrderError(TokenisationError):
    pass


class ObservationOutsideExtentError(TokenisationError):
    pass


class ChannelKindMismatchError(TokenisationError):
    pass


class InvalidArchivedCorpusError(CatalogError, ValueError):
    pass


class MalformedManifestError(CatalogError):
    pass
