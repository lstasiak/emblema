from dataclasses import dataclass

from emblema.serving.application.use_cases.promote_artifact import PromoteArtifact
from emblema.serving.application.use_cases.withdraw_served_model import WithdrawServedModel


@dataclass(frozen=True)
class Services:
    """The use cases this command line can run: put an artifact into service, take one out."""

    promote_artifact: PromoteArtifact
    withdraw_served_model: WithdrawServedModel
