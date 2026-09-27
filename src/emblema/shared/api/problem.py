from http import HTTPStatus
from typing import Any

from pydantic import BaseModel, Field


class Problem(BaseModel):
    """The one shape every error answers in: a problem details document (RFC 9457).

    A client reads one thing whatever went wrong — a model not found, a window too long, a body
    that does not parse — and renders it without knowing which layer refused.
    """

    type: str = Field(
        default="about:blank",
        description="Identifies the kind of problem; the blank one says the status says it all.",
    )
    title: str = Field(description="What went wrong, in a few words that never change.")
    status: int = Field(description="The HTTP status, repeated so the body stands on its own.")
    detail: str | None = Field(default=None, description="What went wrong this time.")
    instance: str | None = Field(default=None, description="The request path the problem is about.")

    @classmethod
    def responses(cls, *statuses: HTTPStatus) -> dict[int | str, dict[str, Any]]:
        """What a route declares it may answer with, for the schema: a problem per status."""
        return {status.value: {"model": cls} for status in statuses}
