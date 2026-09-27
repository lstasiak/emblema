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


class InvalidModelInputError(ServingError, ValueError):
    """A description of what a model takes names no corpus, no channel, or no window length."""


class InvalidAdmittedWindowError(ServingError, ValueError):
    """An admitted window reports a channel as ignored that it still holds a reading on."""


class InvalidInferenceLimitsError(ServingError, ValueError):
    """A limit on what one request may ask is not a positive count."""


class ServedModelNotServingError(ServingError):
    """The model asked to answer has been withdrawn."""


class UnobservedWindowError(ServingError):
    """After the channels the model does not know were dropped, a window holds no reading."""


class WindowTooLongError(ServingError):
    """A window tokenises to more tokens than the service admits."""


class TooManyWindowsError(ServingError):
    """A request asks for more windows than the service admits at once."""


class EmptyRequestError(ServingError):
    """A request carries no window to answer."""


class UntokenisableRequestError(ServingError):
    """The Catalog refused to tokenise a window: a reading of the wrong kind for its channel."""


class UnreadableServedArtifactError(ServingError):
    """The artifact a model serves is not a kept candidate this runtime reads."""


class UnservableArtifactError(ServingError):
    """The kept candidate is stored in no form this runtime runs."""


class EmbeddingUnavailableError(ServingError):
    """The served candidate has no representation to hand out: it is not a network."""


class NonFiniteAnswerError(ServingError):
    """A served model answered a window with a value that is not a finite number."""


class InvalidInferenceBudgetError(ServingError, ValueError):
    """A budget for running windows at once is not a positive count of windows and tokens."""


class WindowBeyondBudgetError(ServingError):
    """A window costs more to run than the budget lets any one batch cost."""


class InferenceBusyError(ServingError):
    """The service is running as much as its budget allows and could not admit more in time."""
