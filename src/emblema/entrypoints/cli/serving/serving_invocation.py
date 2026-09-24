from dataclasses import dataclass

from emblema.serving.application.use_cases.promote_artifact import PromoteArtifactCommand
from emblema.serving.application.use_cases.withdraw_served_model import (
    WithdrawServedModelCommand,
)


@dataclass(frozen=True, kw_only=True)
class ServingInvocation:
    """One run of the serving command line: the command its arguments settle whole.

    Attributes:
        command: What the invocation asks the context to do.
    """

    command: PromoteArtifactCommand | WithdrawServedModelCommand
