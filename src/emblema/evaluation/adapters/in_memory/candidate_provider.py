from collections.abc import Callable, Sequence

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.exceptions import (
    CandidateNotRetainableError,
    UnknownCandidateError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.transfer.unit_error import UnitError
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore

type StatedErrors = Callable[[CampaignCell], Sequence[float]]
"""What a candidate gets wrong on each unit of a cell, in unit order."""


class InMemoryCandidateProvider:
    """Answers with errors stated up front, so a whole campaign runs without arithmetic.

    Nothing is learnt here: what each cell reports is decided by the function the provider was
    given, one error per unit. That is enough to exercise everything around the candidates — the
    grid expanding, cells recording, a campaign refusing to finish early, the comparison reading
    what ran — in the time it takes to call a function, and it is the only way to write those
    tests against known numbers rather than against whatever a model happened to produce.
    """

    def __init__(
        self,
        candidates: Sequence[CampaignCandidate],
        units: Sequence[UnitKey],
        errors: StatedErrors,
        store: ArtifactStore | None = None,
    ) -> None:
        """Answer for ``candidates`` over ``units``, reporting what ``errors`` says.

        Args:
            candidates: Who this provider supplies.
            units: The units every cell is scored over. Held in name order, the one a real
                run reports its errors in, so that two candidates always pair.
            errors: What a cell gets wrong on each unit, one error per unit.
            store: Where a candidate this provider is asked to keep is put; a provider never
                asked to keep one needs none.
        """
        self._candidates = tuple(candidates)
        self._units = tuple(sorted(units, key=str))
        self._errors = errors
        self._store = store

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        for known in self._candidates:
            if known.ref == candidate:
                return known
        raise UnknownCandidateError(f"this provider supplies no candidate {candidate}")

    def evaluate(self, request: CandidateEvaluation) -> CellResult:
        """Answer one cell with the errors this provider was told to report.

        Raises:
            UnknownCandidateError: If this provider supplies no candidate of that name.
            CandidateMismatchError: If the campaign recorded the candidate as anything other
                than what this provider states for it.
        """
        request.declared.must_match(self.describe(request.cell.candidate))
        errors = self._errors(request.cell)
        return CellResult(
            cell=request.cell,
            errors=tuple(
                UnitError(unit=unit, squared_error=error**2, windows=1)
                for unit, error in zip(self._units, errors, strict=True)
            ),
            seconds=0.0,
            artifact=self._kept(request.cell) if request.retain else None,
        )

    def _kept(self, cell: CampaignCell) -> ArtifactRef:
        """The cell's answers, stored, so a retained cell names bytes that really exist.

        Raises:
            CandidateNotRetainableError: If the provider was given nowhere to keep them.
        """
        if self._store is None:
            raise CandidateNotRetainableError(
                "this provider was asked to keep what it fitted and was given no store"
            )
        return self._store.put(repr(tuple(self._errors(cell))).encode())
