from typing import Any

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.scoring.unit_error import UnitError
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm


class CellResultDocument:
    """Reads what a cell produced to a document and back, error per unit included.

    The errors travel as the sums they are, not as a figure derived from them, because they are
    what a paired comparison resamples; a JSON number reads back as the float that was written,
    so a cell run elsewhere is recorded exactly as it was measured.
    """

    def encode(self, result: CellResult) -> dict[str, Any]:
        return {
            "cell": self.encode_cell(result.cell),
            "errors": [
                {"unit": str(e.unit), "squared_error": e.squared_error, "windows": e.windows}
                for e in result.errors
            ],
            "seconds": result.seconds,
            "artifact": None
            if result.artifact is None
            else {
                "key": result.artifact.key,
                "algorithm": str(result.artifact.checksum.algorithm),
                "digest": result.artifact.checksum.digest,
            },
        }

    def decode(self, document: dict[str, Any]) -> CellResult:
        """The result that document holds.

        Raises:
            KeyError: If the document is missing a part of a result.
            ValueError: If what it holds is not a result that stands up.
        """
        artifact = document["artifact"]
        return CellResult(
            cell=self.decode_cell(document["cell"]),
            errors=tuple(
                UnitError(
                    unit=UnitKey(error["unit"]),
                    squared_error=float(error["squared_error"]),
                    windows=error["windows"],
                )
                for error in document["errors"]
            ),
            seconds=float(document["seconds"]),
            artifact=None
            if artifact is None
            else ArtifactRef(
                artifact["key"], Checksum(HashAlgorithm(artifact["algorithm"]), artifact["digest"])
            ),
        )

    @staticmethod
    def encode_cell(cell: CampaignCell) -> dict[str, Any]:
        """A cell's coordinates, as every document that names a cell writes them."""
        return {"candidate": str(cell.candidate), "budget": cell.budget.text(), "seed": cell.seed}

    @staticmethod
    def decode_cell(document: dict[str, Any]) -> CampaignCell:
        return CampaignCell(
            candidate=CandidateRef(document["candidate"]),
            budget=LabelBudget.parse(document["budget"]),
            seed=document["seed"],
        )
