from datetime import timedelta
from uuid import UUID

import pytest

from emblema.serving.adapters.in_memory.served_model_listing import InMemoryServedModelListing
from emblema.serving.adapters.in_memory.served_model_repository import (
    InMemoryServedModelRepository,
)
from emblema.serving.application.use_cases.list_served_models import (
    ListServedModels,
    ListServedModelsQuery,
)
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.domain.served_model_state import ServedModelState
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.exceptions import InvalidCursorError, InvalidPageError
from emblema.shared.kernel.paging.cursor import Cursor
from emblema.shared.kernel.timestamps import UtcDateTime
from tests.serving.support import PROMOTED, WITHDRAWN, served


def listing(*models: ServedModelId) -> ListServedModels:
    repository = InMemoryServedModelRepository()
    for number in range(1, len(models) + 1):
        stored = served(
            served_model_id=ServedModelId(UUID(int=number)),
            artifact=ArtifactRef(
                key=f"durable/{number}", checksum=Checksum.of_bytes(bytes([number]))
            ),
            promoted_at=UtcDateTime(PROMOTED.value + timedelta(hours=number)),
        )
        repository.save(stored.withdraw(WITHDRAWN) if number == 2 else stored)
    return ListServedModels(InMemoryServedModelListing(repository))


THREE = (ServedModelId(UUID(int=1)), ServedModelId(UUID(int=2)), ServedModelId(UUID(int=3)))


def test_models_are_listed_newest_promotion_first_and_a_short_list_ends_the_page() -> None:
    page = listing(*THREE)(ListServedModelsQuery(limit=10))

    assert [m.served_model_id for m in page.items] == [THREE[2], THREE[1], THREE[0]]
    assert page.next_cursor is None


def test_a_page_names_the_next_one_and_the_pages_together_are_the_list() -> None:
    first = listing(*THREE)(ListServedModelsQuery(limit=2))
    assert first.next_cursor is not None
    second = listing(*THREE)(ListServedModelsQuery(after=first.next_cursor, limit=2))

    assert [m.served_model_id for m in first.items] == [THREE[2], THREE[1]]
    assert [m.served_model_id for m in second.items] == [THREE[0]]
    assert second.next_cursor is None


def test_a_list_can_be_narrowed_to_the_models_in_service() -> None:
    page = listing(*THREE)(ListServedModelsQuery(limit=10, state=ServedModelState.SERVING))

    assert [m.served_model_id for m in page.items] == [THREE[2], THREE[0]]


@pytest.mark.parametrize(
    "parts",
    [
        ("a", "b", "c"),
        (str(THREE[0]),),
        ("not-a-time", str(THREE[0])),
        ("2026-01-01T00:00:00", str(THREE[0])),
        (PROMOTED.value.isoformat(), "not-a-uuid"),
    ],
)
def test_a_cursor_this_list_did_not_issue_is_refused(parts: tuple[str, ...]) -> None:
    with pytest.raises(InvalidCursorError):
        listing(*THREE)(ListServedModelsQuery(after=Cursor(parts), limit=2))


def test_a_page_continues_after_its_position_whether_or_not_that_model_is_still_stored() -> None:
    position = UtcDateTime(PROMOTED.value + timedelta(hours=3)).value.isoformat()
    after_nobody = Cursor((position, str(UUID(int=99))))

    page = listing(*THREE)(ListServedModelsQuery(after=after_nobody, limit=5))

    assert [m.served_model_id for m in page.items] == [THREE[1], THREE[0]]


def test_a_page_holds_at_least_one_model() -> None:
    with pytest.raises(InvalidPageError):
        ListServedModelsQuery(limit=0)
