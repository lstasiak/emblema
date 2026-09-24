import pytest

from emblema.serving.adapters.in_memory.served_model_repository import (
    InMemoryServedModelRepository,
)
from emblema.serving.application.use_cases.withdraw_served_model import (
    WithdrawServedModel,
    WithdrawServedModelCommand,
)
from emblema.serving.domain.exceptions import ServedModelNotFoundError, ServedModelWithdrawnError
from emblema.serving.domain.served_model_state import ServedModelState
from emblema.shared.adapters.in_memory.clock import FixedClock
from tests.serving.support import MODEL, WITHDRAWN, served


def withdrawing(stored: bool = True) -> tuple[WithdrawServedModel, InMemoryServedModelRepository]:
    models = InMemoryServedModelRepository()
    if stored:
        models.save(served())
    return WithdrawServedModel(models, FixedClock(WITHDRAWN)), models


def test_a_withdrawn_model_is_kept_out_of_service_from_now() -> None:
    withdraw, models = withdrawing()

    withdraw(WithdrawServedModelCommand(served_model=MODEL))

    assert models.get(MODEL).state is ServedModelState.WITHDRAWN
    assert models.get(MODEL).withdrawn_at == WITHDRAWN


def test_a_model_withdrawn_once_is_not_withdrawn_again() -> None:
    withdraw, _ = withdrawing()
    withdraw(WithdrawServedModelCommand(served_model=MODEL))

    with pytest.raises(ServedModelWithdrawnError):
        withdraw(WithdrawServedModelCommand(served_model=MODEL))


def test_a_model_nobody_promoted_is_not_withdrawn() -> None:
    withdraw, _ = withdrawing(stored=False)

    with pytest.raises(ServedModelNotFoundError):
        withdraw(WithdrawServedModelCommand(served_model=MODEL))
