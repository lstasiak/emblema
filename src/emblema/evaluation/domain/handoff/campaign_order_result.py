from dataclasses import dataclass, replace
from typing import Self

from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.exceptions import InvalidCampaignOrderResultError
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class CampaignOrderResult:
    """What a machine has answered of an order so far, to be recorded where the campaign lives.

    Reported again after every cell, each report holding every cell answered until then, so a
    session that is cut short leaves behind everything it finished under the last reference it
    logged. The order it answers is named by reference, which names one document for good, so
    the result is held to the order it was run for and not to one written afterwards.

    Invariants: at least one cell answered, none twice; the commit is non-empty without
    surrounding whitespace.

    Attributes:
        order: The order this answers.
        campaign: The campaign the cells belong to.
        git_commit: Revision of the code the cells were run with.
        results: Each answered cell, in the order it was run.
    """

    order: ArtifactRef
    campaign: CampaignId
    git_commit: str
    results: tuple[CellResult, ...]

    def __post_init__(self) -> None:
        if not self.results:
            raise InvalidCampaignOrderResultError("a result of an order answers at least one cell")
        cells = [result.cell for result in self.results]
        if len(set(cells)) != len(cells):
            raise InvalidCampaignOrderResultError("a result of an order answers a cell twice")
        if not self.git_commit or self.git_commit != self.git_commit.strip():
            raise InvalidCampaignOrderResultError(
                "git_commit must be non-empty without surrounding whitespace"
            )

    @property
    def cells(self) -> frozenset[CampaignCell]:
        return frozenset(result.cell for result in self.results)

    def with_result(self, result: CellResult) -> Self:
        """This report with one more cell answered.

        Raises:
            InvalidCampaignOrderResultError: If that cell was already answered.
        """
        return replace(self, results=(*self.results, result))
