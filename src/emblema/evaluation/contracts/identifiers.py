"""The identities other contexts refer Evaluation's work by."""

from dataclasses import dataclass

from emblema.evaluation.contracts.exceptions import InvalidCandidateRefError
from emblema.shared.kernel.identifiers import EntityId


@dataclass(frozen=True)
class TaskId(EntityId):
    """Identity of a downstream task, the Evaluation entity other contexts refer to.

    It lives in the published language because a campaign's outcome names the task it was run
    for, and whoever reads that outcome must be able to say which task it was without reaching
    into this context.
    """


@dataclass(frozen=True)
class CampaignId(EntityId):
    """Identity of an evaluation campaign.

    Published because the promise Serving relies on — that an artifact was measured in a
    campaign that finished — is worth only as much as the ability to name which campaign.
    """


@dataclass(frozen=True)
class CandidateRef:
    """What one competitor in a campaign is called, in words rather than a generated identity.

    A candidate is named by whoever designs the campaign, and the name is what the grid, the
    reports and the outcome all refer to, so it is text chosen to read well rather than a UUID.
    Two campaigns may use the same name for the same thing on purpose: that is how a curve
    measured twice is recognised as the same arm measured twice.

    Attributes:
        value: Non-blank text without surrounding whitespace.
    """

    value: str

    def __post_init__(self) -> None:
        if not self.value or self.value != self.value.strip():
            raise InvalidCandidateRefError(
                "a candidate reference must be non-blank without surrounding whitespace"
            )

    def __str__(self) -> str:
        return self.value
