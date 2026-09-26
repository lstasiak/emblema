import logging
from collections.abc import Callable, Sequence
from http import HTTPStatus

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from opentelemetry import trace
from starlette.exceptions import HTTPException

from emblema.shared.api.problem import Problem
from emblema.shared.kernel.exceptions import (
    InvalidCursorError,
    InvalidEntityIdError,
    InvalidPageError,
)

MEDIA_TYPE = "application/problem+json"

logger = logging.getLogger(__name__)

# What the shared kernel refuses on the way in: a cursor, a page or an identity a client wrote.
_KERNEL: tuple[tuple[type[Exception], HTTPStatus], ...] = (
    (InvalidCursorError, HTTPStatus.UNPROCESSABLE_ENTITY),
    (InvalidPageError, HTTPStatus.UNPROCESSABLE_ENTITY),
    (InvalidEntityIdError, HTTPStatus.UNPROCESSABLE_ENTITY),
)


class ProblemDetails:
    """Answers every refusal and every failure in one shape, whichever layer refused.

    Each context states what it refuses and with which status; this registers those beside the
    kernel's and the framework's, and a failure nobody named answers in the same shape, so there
    is one error format and not two. A client error carries its reason. A server error carries
    none: the reason is logged with the trace the request ran under, and the body names that
    trace, so whoever reads the log can find it and whoever calls the service learns nothing
    about its insides.
    """

    def __init__(self, refusals: Sequence[tuple[type[Exception], HTTPStatus]]) -> None:
        self._refusals = (*_KERNEL, *refusals)

    def register(self, app: FastAPI) -> None:
        app.add_exception_handler(RequestValidationError, self._validation)
        app.add_exception_handler(HTTPException, self._http)
        for refusal, status in self._refusals:
            app.add_exception_handler(refusal, self._refused(status))
        app.add_exception_handler(Exception, self._refused(HTTPStatus.INTERNAL_SERVER_ERROR))

    @staticmethod
    def problem(request: Request, status: HTTPStatus, detail: str) -> JSONResponse:
        """The problem document for ``status``, about the request's path."""
        body = Problem(
            title=status.phrase, status=status.value, detail=detail, instance=request.url.path
        )
        return JSONResponse(
            status_code=status.value, content=body.model_dump(), media_type=MEDIA_TYPE
        )

    def _refused(self, status: HTTPStatus) -> Callable[[Request, Exception], Response]:
        def handler(request: Request, error: Exception) -> Response:
            if status < HTTPStatus.INTERNAL_SERVER_ERROR:
                return self.problem(request, status, str(error))
            logger.error("%s %s failed", request.method, request.url.path, exc_info=error)
            return self.problem(request, status, self._failure())

        return handler

    @staticmethod
    def _failure() -> str:
        context = trace.get_current_span().get_span_context()
        if not context.is_valid:
            return "the service failed to answer; the failure is logged"
        return (
            "the service failed to answer; the failure is logged under trace "
            f"{context.trace_id:032x}"
        )

    def _validation(self, request: Request, error: Exception) -> Response:
        detail = str(error)
        if isinstance(error, RequestValidationError):
            detail = "; ".join(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in error.errors()
            )
        return self.problem(request, HTTPStatus.UNPROCESSABLE_ENTITY, detail)

    def _http(self, request: Request, error: Exception) -> Response:
        if isinstance(error, HTTPException):
            return self.problem(request, HTTPStatus(error.status_code), str(error.detail))
        # Registered for the framework's exception alone; the branch keeps the type honest.
        return self.problem(
            request, HTTPStatus.INTERNAL_SERVER_ERROR, str(error)
        )  # pragma: no cover
