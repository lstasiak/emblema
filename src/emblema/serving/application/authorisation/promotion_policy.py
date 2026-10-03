from typing import ClassVar

from emblema.serving.domain.exceptions import OperationNotPermittedError
from emblema.shared.kernel.identity.principal import Principal
from emblema.shared.kernel.identity.scope import Scope


class PromotionPolicy:
    """Who may change what is served.

    A rule of the project, not of the transport: the use case asks it of the principal the edge
    identified, whichever edge that was, so the command line and the HTTP service refuse the
    same callers for the same reason, and the rule is tested without either. One scope per
    operation: taking a model out of service does not follow from being allowed to put one in.

    Attributes:
        PROMOTE: The scope that allows putting an artifact into service.
        WITHDRAW: The scope that allows taking a served model out of service.
    """

    PROMOTE: ClassVar[Scope] = Scope("serving:promote")
    WITHDRAW: ClassVar[Scope] = Scope("serving:withdraw")

    def permit_promotion(self, actor: Principal) -> None:
        """Let the actor promote, or refuse.

        Raises:
            OperationNotPermittedError: If the actor was not granted promotion.
        """
        self._permit(actor, self.PROMOTE, "promote an artifact")

    def permit_withdrawal(self, actor: Principal) -> None:
        """Let the actor withdraw, or refuse.

        Raises:
            OperationNotPermittedError: If the actor was not granted withdrawal.
        """
        self._permit(actor, self.WITHDRAW, "withdraw a served model")

    @staticmethod
    def _permit(actor: Principal, scope: Scope, operation: str) -> None:
        if not actor.permits(scope):
            raise OperationNotPermittedError(
                f"{actor.subject} may not {operation}: scope {scope} was not granted"
            )
