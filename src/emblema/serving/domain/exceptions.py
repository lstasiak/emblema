"""Every exception the Serving context raises, including those its ports declare."""


class ServingError(Exception):
    """Base of everything this context raises."""


class InvalidCampaignScoreError(ServingError, ValueError):
    """A score is unnamed, counts no window, is not finite, or stands on no repeat."""


class InvalidPromotableArtifactError(ServingError, ValueError):
    """A promotable artifact states the same operating point twice."""


class InvalidServedModelError(ServingError, ValueError):
    """A served model predates the campaign it came from, or was withdrawn before promotion."""


class ArtifactNotPromotableError(ServingError):
    """No finished campaign kept an artifact with that checksum."""


class AmbiguousArtifactError(ServingError):
    """Several finished campaigns kept an artifact with that checksum, and none was named."""


class ArtifactUnavailableError(ServingError):
    """The artifact a campaign kept is no longer in the store, so there is nothing to serve."""


class ArtifactAlreadyServedError(ServingError):
    """The artifact is already served by a model that has not been withdrawn."""


class ServedModelNotFoundError(ServingError):
    """No served model is stored under that identity."""


class ServedModelWithdrawnError(ServingError):
    """A served model that had already been withdrawn was asked to be withdrawn again."""
