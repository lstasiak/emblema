from datetime import UTC, datetime

import pytest

from emblema.serving.domain.exceptions import InvalidServedModelError, ServedModelWithdrawnError
from emblema.serving.domain.served_model import ServedModel
from emblema.serving.domain.served_model_state import ServedModelState
from emblema.shared.kernel.timestamps import UtcDateTime
from tests.serving.support import FINISHED, MODEL, PROMOTED, WITHDRAWN, promotable, served

BEFORE_FINISHING = UtcDateTime(datetime(2026, 8, 31, tzinfo=UTC))


def test_a_promoted_model_serves_exactly_what_the_campaign_kept() -> None:
    kept = promotable()

    model = ServedModel.promoted(kept, served_model_id=MODEL, at=PROMOTED)

    assert (model.origin, model.kind, model.artifact) == (kept.origin, kept.kind, kept.artifact)
    assert model.state is ServedModelState.SERVING


def test_a_model_is_not_promoted_before_its_campaign_finished() -> None:
    with pytest.raises(InvalidServedModelError, match="had not finished"):
        ServedModel.promoted(promotable(), served_model_id=MODEL, at=BEFORE_FINISHING)


def test_a_model_may_be_promoted_the_moment_its_campaign_finished() -> None:
    assert ServedModel.promoted(promotable(), served_model_id=MODEL, at=FINISHED).promoted_at == (
        FINISHED
    )


def test_a_withdrawn_model_is_out_of_service_from_the_moment_it_was_withdrawn() -> None:
    withdrawn = served().withdraw(WITHDRAWN)

    assert withdrawn.state is ServedModelState.WITHDRAWN
    assert withdrawn.withdrawn_at == WITHDRAWN


def test_a_withdrawn_model_is_not_withdrawn_again() -> None:
    with pytest.raises(ServedModelWithdrawnError):
        served().withdraw(WITHDRAWN).withdraw(WITHDRAWN)


def test_a_model_is_not_withdrawn_before_it_was_promoted() -> None:
    with pytest.raises(InvalidServedModelError, match="before it was promoted"):
        served().withdraw(FINISHED)
