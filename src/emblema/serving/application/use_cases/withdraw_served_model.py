from dataclasses import dataclass

from emblema.serving.application.authorisation.promotion_policy import PromotionPolicy
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.ports.served_model_repository import ServedModelRepository
from emblema.shared.kernel.identity.principal import Principal
from emblema.shared.ports.clock import Clock


@dataclass(frozen=True, kw_only=True)
class WithdrawServedModelCommand:
    """Request to take a served model out of service.

    Attributes:
        actor: Who asks, as the edge identified them; whether they may is decided here.
        served_model: Which model to withdraw.
    """

    actor: Principal
    served_model: ServedModelId


class WithdrawServedModel:
    """Takes a served model out of service, keeping the record that it served."""

    def __init__(
        self, served: ServedModelRepository, clock: Clock, policy: PromotionPolicy
    ) -> None:
        self._served = served
        self._clock = clock
        self._policy = policy

    def __call__(self, command: WithdrawServedModelCommand) -> None:
        """Withdraw the model now.

        Raises:
            OperationNotPermittedError: If the actor was not granted withdrawal.
            ServedModelNotFoundError: If no model is stored under that identity.
            ServedModelWithdrawnError: If it was already withdrawn.
        """
        self._policy.permit_withdrawal(command.actor)
        model = self._served.get(command.served_model)
        self._served.save(model.withdraw(self._clock.now()))
