from dataclasses import dataclass
from datetime import datetime

from emblema.serving.application.read_models.served_model_summary import ServedModelSummary
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.domain.served_model_position import ServedModelPosition
from emblema.serving.domain.served_model_state import ServedModelState
from emblema.serving.ports.served_model_listing import ServedModelListing
from emblema.shared.kernel.exceptions import InvalidCursorError, InvalidPageError
from emblema.shared.kernel.paging.cursor import Cursor
from emblema.shared.kernel.paging.page import Page
from emblema.shared.kernel.timestamps import UtcDateTime


@dataclass(frozen=True, kw_only=True)
class ListServedModelsQuery:
    """Which page of served models to show.

    Invariants: the limit is positive.

    Attributes:
        after: Where the previous page ended; ``None`` for the first page.
        limit: How many models the page holds at most.
        state: Only models in this state, or every model where ``None``.
    """

    after: Cursor | None = None
    limit: int
    state: ServedModelState | None = None

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise InvalidPageError(f"a page holds at least one model, got a limit of {self.limit}")


class ListServedModels:
    """Lists served models newest first, a page at a time, each page naming the next.

    The cursor carries the position of the last model shown — when it was promoted and its
    identity — so the next page is read after it without looking the model up; one more model
    than the page holds is read, so that the page knows whether there is a next one without a
    count over the table.
    """

    def __init__(self, listing: ServedModelListing) -> None:
        self._listing = listing

    def __call__(self, query: ListServedModelsQuery) -> Page[ServedModelSummary]:
        """The page, and the cursor that opens the next one where there is one.

        Raises:
            InvalidCursorError: If the cursor is not one this issued.
        """
        after = None if query.after is None else self._position(query.after)
        models = self._listing.page(after=after, limit=query.limit + 1, state=query.state)
        shown = models[: query.limit]
        return Page(
            items=tuple(ServedModelSummary.of(model) for model in shown),
            next_cursor=(
                self._cursor(ServedModelPosition.of(shown[-1]))
                if len(models) > query.limit
                else None
            ),
        )

    @staticmethod
    def _cursor(position: ServedModelPosition) -> Cursor:
        return Cursor((position.promoted_at.value.isoformat(), str(position.served_model_id)))

    @staticmethod
    def _position(cursor: Cursor) -> ServedModelPosition:
        promoted_at, served_model_id = cursor.unpack(2)
        try:
            return ServedModelPosition(
                promoted_at=UtcDateTime(datetime.fromisoformat(promoted_at)),
                served_model_id=ServedModelId.parse(served_model_id),
            )
        except ValueError as error:
            raise InvalidCursorError(f"not a cursor of this list: {cursor.parts}") from error
