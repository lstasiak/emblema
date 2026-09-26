from http import HTTPStatus
from typing import ClassVar

from emblema.catalog.contracts.exceptions import CatalogContractError
from emblema.serving.domain.exceptions import (
    ArtifactUnavailableError,
    EmbeddingUnavailableError,
    NonFiniteAnswerError,
    ServedModelNotFoundError,
    ServedModelNotServingError,
    ServingError,
    UnreadableServedArtifactError,
    UnservableArtifactError,
)


class ServingRefusals:
    """What Serving refuses over HTTP, and the status each refusal answers with.

    The status is the one thing the domain does not know about a refusal, so it is decided here,
    once per kind, beside the routes that raise them. The process registers each class on its
    own, and the framework picks the handler of the nearest class in the hierarchy, so a
    specific refusal answers with its own status and every other one falls under the base. A
    window that breaks the Catalog's rules for a window is the caller's error, as it arrives
    through this context's routes.

    Attributes:
        STATUSES: Each refusal and its status, most specific first.
    """

    STATUSES: ClassVar[tuple[tuple[type[Exception], HTTPStatus], ...]] = (
        (ServedModelNotFoundError, HTTPStatus.NOT_FOUND),
        (ServedModelNotServingError, HTTPStatus.CONFLICT),
        (EmbeddingUnavailableError, HTTPStatus.CONFLICT),
        (UnservableArtifactError, HTTPStatus.CONFLICT),
        (ArtifactUnavailableError, HTTPStatus.INTERNAL_SERVER_ERROR),
        (UnreadableServedArtifactError, HTTPStatus.INTERNAL_SERVER_ERROR),
        (NonFiniteAnswerError, HTTPStatus.INTERNAL_SERVER_ERROR),
        (CatalogContractError, HTTPStatus.UNPROCESSABLE_ENTITY),
        (ServingError, HTTPStatus.UNPROCESSABLE_ENTITY),
    )
