from collections.abc import Callable
from http import HTTPStatus
from typing import Annotated, Any, ClassVar

from fastapi import APIRouter, Depends, Query

from emblema.serving.api.schemas.embeddings_resource import EmbeddingsResource
from emblema.serving.api.schemas.inference_request import InferenceRequest
from emblema.serving.api.schemas.predictions_resource import PredictionsResource
from emblema.serving.api.schemas.served_model_detail import ServedModelDetail
from emblema.serving.api.schemas.served_model_resource import ServedModelResource
from emblema.serving.application.read_models.embeddings import Embeddings
from emblema.serving.application.read_models.predictions import Predictions
from emblema.serving.application.read_models.served_model_summary import ServedModelSummary
from emblema.serving.application.read_models.served_model_view import ServedModelView
from emblema.serving.application.use_cases.embed_windows import EmbedWindowsQuery
from emblema.serving.application.use_cases.list_served_models import ListServedModelsQuery
from emblema.serving.application.use_cases.predict_windows import PredictWindowsQuery
from emblema.serving.application.use_cases.view_served_model import ViewServedModelQuery
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.domain.served_model_state import ServedModelState
from emblema.shared.api.cursor_page import CursorPage
from emblema.shared.api.page_request import CursorParameter, LimitParameter, PageRequests
from emblema.shared.api.problem import Problem
from emblema.shared.api.request_rate_limit import RequestRateLimit
from emblema.shared.kernel.paging.page import Page


class ServedModelRoutes:
    """What the API says and does about served models, as one router the process includes.

    Every handler deserialises, calls one use case and serialises. The use cases arrive as the
    callables they are, so the process may wrap one — to count what it answered — without this
    router knowing; the page a list is asked for is read the way every list reads it. The two
    routes that run a model are open to anyone and so are the ones rationed: the process says
    how much, this router says which.

    Attributes:
        router: The routes, under ``/served-models``.
    """

    _REFUSALS: ClassVar[dict[int | str, dict[str, Any]]] = Problem.responses(
        HTTPStatus.NOT_FOUND, HTTPStatus.CONFLICT, HTTPStatus.UNPROCESSABLE_ENTITY
    )
    _INFERENCE_REFUSALS: ClassVar[dict[int | str, dict[str, Any]]] = {
        **_REFUSALS,
        **Problem.responses(HTTPStatus.TOO_MANY_REQUESTS, HTTPStatus.SERVICE_UNAVAILABLE),
    }

    def __init__(
        self,
        *,
        predict: Callable[[PredictWindowsQuery], Predictions],
        embed: Callable[[EmbedWindowsQuery], Embeddings],
        view: Callable[[ViewServedModelQuery], ServedModelView],
        listed: Callable[[ListServedModelsQuery], Page[ServedModelSummary]],
        pages: PageRequests,
        throttle: RequestRateLimit,
    ) -> None:
        self._predict = predict
        self._embed = embed
        self._view = view
        self._listed = listed
        self._pages = pages
        self.router = APIRouter(prefix="/served-models")
        self.router.get(
            "",
            operation_id="list_served_models",
            summary="Served models, newest promotion first",
            response_model=CursorPage[ServedModelResource],
            responses=Problem.responses(HTTPStatus.UNPROCESSABLE_ENTITY),
            tags=["served models"],
        )(self.list_models)
        self.router.get(
            "/{served_model_id}",
            operation_id="view_served_model",
            summary="One served model and what it takes as input",
            response_model=ServedModelDetail,
            responses=self._REFUSALS,
            tags=["served models"],
        )(self.view_model)
        self.router.post(
            "/{served_model_id}/predictions",
            operation_id="predict",
            summary="Answer windows of raw readings with a served model",
            response_model=PredictionsResource,
            responses=self._INFERENCE_REFUSALS,
            tags=["inference"],
            dependencies=[Depends(throttle)],
        )(self.predict)
        self.router.post(
            "/{served_model_id}/embeddings",
            operation_id="embed",
            summary="Represent windows of raw readings with a served model",
            response_model=EmbeddingsResource,
            responses=self._INFERENCE_REFUSALS,
            tags=["inference"],
            dependencies=[Depends(throttle)],
        )(self.embed)

    def list_models(
        self,
        cursor: CursorParameter = None,
        limit: LimitParameter = None,
        state: Annotated[
            ServedModelState | None, Query(description="Only models in this state.")
        ] = None,
    ) -> CursorPage[ServedModelResource]:
        page = self._pages.read(cursor, limit)
        listed = self._listed(
            ListServedModelsQuery(after=page.after, limit=page.limit, state=state)
        )
        return CursorPage[ServedModelResource].of(listed, ServedModelResource.of)

    def view_model(self, served_model_id: str) -> ServedModelDetail:
        view = self._view(ViewServedModelQuery(served_model=ServedModelId.parse(served_model_id)))
        return ServedModelDetail.of(view)

    def predict(self, served_model_id: str, request: InferenceRequest) -> PredictionsResource:
        predictions = self._predict(
            PredictWindowsQuery(
                served_model=ServedModelId.parse(served_model_id),
                windows=tuple(window.to_message() for window in request.windows),
            )
        )
        return PredictionsResource.of(predictions)

    def embed(self, served_model_id: str, request: InferenceRequest) -> EmbeddingsResource:
        embeddings = self._embed(
            EmbedWindowsQuery(
                served_model=ServedModelId.parse(served_model_id),
                windows=tuple(window.to_message() for window in request.windows),
            )
        )
        return EmbeddingsResource.of(embeddings)
