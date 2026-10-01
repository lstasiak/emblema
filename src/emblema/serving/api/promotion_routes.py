from collections.abc import Callable
from http import HTTPStatus
from typing import Any, ClassVar

from fastapi import APIRouter, Depends, Request, Response

from emblema.serving.api.schemas.promoted_model_resource import PromotedModelResource
from emblema.serving.api.schemas.promotion_request import PromotionRequest
from emblema.serving.application.use_cases.promote_artifact import PromoteArtifactCommand
from emblema.serving.application.use_cases.withdraw_served_model import (
    WithdrawServedModelCommand,
)
from emblema.serving.domain.identifiers import ServedModelId
from emblema.shared.api.bearer_authentication import BearerAuthentication
from emblema.shared.api.problem import Problem


class PromotionRoutes:
    """What changes which models are served, as one router every route of which is protected.

    A router apart from the one that answers requests, so that the line between what is public
    and what needs a token is a line between routers, not a flag on each route. The router
    identifies the caller once, before any handler runs; a handler reads the actor back and
    puts it in the command, and the use case decides whether that actor may.

    Attributes:
        router: The routes, under ``/served-models``.
    """

    _REFUSALS: ClassVar[dict[int | str, dict[str, Any]]] = Problem.responses(
        HTTPStatus.UNAUTHORIZED,
        HTTPStatus.FORBIDDEN,
        HTTPStatus.NOT_FOUND,
        HTTPStatus.CONFLICT,
        HTTPStatus.UNPROCESSABLE_ENTITY,
    )

    def __init__(
        self,
        *,
        promote: Callable[[PromoteArtifactCommand], ServedModelId],
        withdraw: Callable[[WithdrawServedModelCommand], None],
        authentication: BearerAuthentication,
    ) -> None:
        self._promote = promote
        self._withdraw = withdraw
        self._authentication = authentication
        self.router = APIRouter(
            prefix="/served-models",
            tags=["promotion"],
            dependencies=[Depends(authentication)],
        )
        self.router.post(
            "",
            operation_id="promote",
            summary="Put an artifact a finished campaign kept into service",
            status_code=HTTPStatus.CREATED.value,
            response_model=PromotedModelResource,
            responses=self._REFUSALS,
        )(self.promote)
        self.router.post(
            "/{served_model_id}/withdrawal",
            operation_id="withdraw",
            summary="Take a served model out of service, keeping the record that it served",
            status_code=HTTPStatus.NO_CONTENT.value,
            response_class=Response,
            responses=self._REFUSALS,
        )(self.withdraw)

    def promote(
        self, request: Request, body: PromotionRequest, response: Response
    ) -> PromotedModelResource:
        promoted = self._promote(body.to_command(self._authentication.read_actor(request)))
        response.headers["Location"] = f"{self.router.prefix}/{promoted}"
        return PromotedModelResource.of(promoted)

    def withdraw(self, request: Request, served_model_id: str) -> Response:
        self._withdraw(
            WithdrawServedModelCommand(
                actor=self._authentication.read_actor(request),
                served_model=ServedModelId.parse(served_model_id),
            )
        )
        return Response(status_code=HTTPStatus.NO_CONTENT.value)
