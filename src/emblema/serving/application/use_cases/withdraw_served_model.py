from dataclasses import dataclass

from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.ports.served_model_repository import ServedModelRepository
from emblema.shared.ports.clock import Clock


@dataclass(frozen=True, kw_only=True)
class WithdrawServedModelCommand:
    """Request to take a served model out of service.

    Attributes:
        served_model: Which model to withdraw.
    """

    served_model: ServedModelId


class WithdrawServedModel:
    """Takes a served model out of service, keeping the record that it served."""

    def __init__(self, served: ServedModelRepository, clock: Clock) -> None:
        self._served = served
        self._clock = clock

    def __call__(self, command: WithdrawServedModelCommand) -> None:
        """Withdraw the model now.

        Raises:
            ServedModelNotFoundError: If no model is stored under that identity.
            ServedModelWithdrawnError: If it was already withdrawn.
        """
        model = self._served.get(command.served_model)
        self._served.save(model.withdraw(self._clock.now()))
