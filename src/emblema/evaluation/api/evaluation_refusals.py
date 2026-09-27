from http import HTTPStatus
from typing import ClassVar

from emblema.evaluation.domain.exceptions import CampaignNotFoundError, EvaluationError


class EvaluationRefusals:
    """What Evaluation refuses over HTTP, and the status each refusal answers with.

    Decided here, beside the routes that raise them, for the reason Serving decides its own: the
    status is the one thing the domain does not know about a refusal.

    Attributes:
        STATUSES: Each refusal and its status, most specific first.
    """

    STATUSES: ClassVar[tuple[tuple[type[Exception], HTTPStatus], ...]] = (
        (CampaignNotFoundError, HTTPStatus.NOT_FOUND),
        (EvaluationError, HTTPStatus.UNPROCESSABLE_ENTITY),
    )
