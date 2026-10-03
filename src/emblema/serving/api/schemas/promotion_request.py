from pydantic import BaseModel, Field, field_validator

from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.serving.application.use_cases.promote_artifact import PromoteArtifactCommand
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.identity.principal import Principal


class PromotionRequest(BaseModel):
    """Which kept artifact to put into service, named as the campaign's announcement names it."""

    checksum: str = Field(
        description="What the artifact's content hashes to, written algorithm:digest."
    )
    campaign_id: str | None = Field(
        default=None, description="The campaign to promote it out of, where several kept it."
    )
    candidate: str | None = Field(
        default=None,
        description="The competitor to promote it as, where one campaign kept it as several.",
    )

    @field_validator("checksum")
    @classmethod
    def _a_checksum(cls, value: str) -> str:
        Checksum.parse(value)
        return value

    @field_validator("campaign_id")
    @classmethod
    def _a_campaign_id(cls, value: str | None) -> str | None:
        if value is not None:
            CampaignId.parse(value)
        return value

    @field_validator("candidate")
    @classmethod
    def _a_candidate(cls, value: str | None) -> str | None:
        if value is not None:
            CandidateRef(value)
        return value

    def to_command(self, actor: Principal) -> PromoteArtifactCommand:
        """The command, on behalf of the caller the edge identified."""
        return PromoteArtifactCommand(
            actor=actor,
            checksum=Checksum.parse(self.checksum),
            campaign=None if self.campaign_id is None else CampaignId.parse(self.campaign_id),
            candidate=None if self.candidate is None else CandidateRef(self.candidate),
        )
