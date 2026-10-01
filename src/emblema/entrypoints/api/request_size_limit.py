import json
from http import HTTPStatus
from typing import ClassVar

from starlette.types import ASGIApp, Receive, Scope, Send

from emblema.entrypoints.api.problem_details import ProblemDetails
from emblema.shared.api.problem import Problem


class RequestSizeLimit:
    """Refuses a request whose body is larger than the service reads, before it is read.

    The framework parses a body whole before any route sees it, so a limit on windows or tokens
    comes too late for a body of a gigabyte: the parse alone is the cost. The declared length is
    held to the limit instead, and a body sent without one is refused rather than read to find
    out, which a client that sends JSON never has to do.

    Not the framework's own body limit: that one reads a body sent without a length to count
    it, and refuses in plain text, where this process answers every refusal in one shape.
    """

    _BODILESS: ClassVar[frozenset[str]] = frozenset({"GET", "HEAD", "OPTIONS", "DELETE"})

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        if max_bytes < 1:
            raise ValueError(f"a request may carry at least one byte, got {max_bytes}")
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] in self._BODILESS:
            await self._app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        declared = headers.get(b"content-length")
        if declared is None:
            if b"transfer-encoding" in headers:
                await self._refuse(
                    scope, send, HTTPStatus.LENGTH_REQUIRED, "a body states its length"
                )
                return
        elif not declared.isdigit() or int(declared) > self._max_bytes:
            await self._refuse(
                scope,
                send,
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"a request carries at most {self._max_bytes} bytes",
            )
            return
        await self._app(scope, receive, send)

    @staticmethod
    async def _refuse(scope: Scope, send: Send, status: HTTPStatus, detail: str) -> None:
        body = json.dumps(
            Problem(
                title=status.phrase, status=status.value, detail=detail, instance=scope["path"]
            ).model_dump()
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status.value,
                "headers": [
                    (b"content-type", ProblemDetails.MEDIA_TYPE.encode("ascii")),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
