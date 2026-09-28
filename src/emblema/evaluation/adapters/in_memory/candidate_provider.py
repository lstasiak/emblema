from collections.abc import Callable, Sequence
from dataclasses import dataclass

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
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.unit_error import UnitError
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore


@dataclass(frozen=True)
class StatedErrors:
    """What a candidate gets wrong on each unit of a cell, in unit order; no answer is kept.

    Attributes:
        of: The error per unit of a cell, in the order the provider holds its units.
    """

    of: Callable[[CampaignCell], Sequence[float]]

    def scored(
        self, cell: CampaignCell, units: Sequence[UnitKey]
    ) -> tuple[tuple[UnitError, ...], tuple[WindowPrediction, ...]]:
        errors = tuple(
            UnitError(unit=unit, squared_error=error**2, windows=1)
            for unit, error in zip(units, self.of(cell), strict=True)
        )
        return errors, ()


@dataclass(frozen=True)
class StatedAnswers:
    """What each unit of a cell holds and what the candidate answers for it, in unit order.

    The cell keeps the answers, one window per unit, and its errors are theirs.

    Attributes:
        of: The ``(target, answer)`` per unit of a cell, in the order the provider holds its
            units.
    """

    of: Callable[[CampaignCell], Sequence[tuple[float, float]]]

    def scored(
        self, cell: CampaignCell, units: Sequence[UnitKey]
    ) -> tuple[tuple[UnitError, ...], tuple[WindowPrediction, ...]]:
        predictions = tuple(
            WindowPrediction(
                window=TaskWindow(unit=unit, position=place, ends_at=0.0),
                target=target,
                predicted=answer,
            )
            for place, (unit, (target, answer)) in enumerate(zip(units, self.of(cell), strict=True))
        )
        return UnitError.per_unit(predictions), predictions


class InMemoryCandidateProvider:
    """Answers with results stated up front, so a whole campaign runs without arithmetic.

    Nothing is learnt here: what each cell reports is decided by the function the provider was
    given, an error per unit or an answer per unit. That is enough to exercise everything around
    the candidates, from the grid expanding to the comparison reading what ran, in the time it
    takes to call a function, against known numbers rather than whatever a model produced.
    """

    def __init__(
        self,
        candidates: Sequence[CampaignCandidate],
        units: Sequence[UnitKey],
        stated: StatedErrors | StatedAnswers,
        store: ArtifactStore | None = None,
    ) -> None:
        """Answer for ``candidates`` over ``units``, reporting what ``stated`` says of each cell.

        Args:
            candidates: Who this provider supplies.
            units: The units every cell is scored over. Held in name order, the one a real
                run reports its errors in, so that two candidates always pair.
            stated: What a cell scores: its errors per unit, or its answers.
            store: Where a candidate this provider is asked to keep is put; a provider never
                asked to keep one needs none.
        """
        self._candidates = tuple(candidates)
        self._units = tuple(sorted(units, key=str))
        self._stated = stated
        self._store = store

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        for known in self._candidates:
            if known.ref == candidate:
                return known
        raise UnknownCandidateError(f"this provider supplies no candidate {candidate}")

    def evaluate(self, request: CandidateEvaluation) -> CellResult:
        """Answer one cell with what this provider was told to report for it.

        Raises:
            UnknownCandidateError: If this provider supplies no candidate of that name.
            CandidateMismatchError: If the campaign recorded the candidate as anything other
                than what this provider states for it.
        """
        request.declared.must_match(self.describe(request.cell.candidate))
        errors, predictions = self._stated.scored(request.cell, self._units)
        return CellResult(
            cell=request.cell,
            errors=errors,
            seconds=0.0,
            artifact=self._kept(request.cell) if request.retain else None,
            predictions=predictions,
        )

    def _kept(self, cell: CampaignCell) -> ArtifactRef:
        """What the cell was told to report, stored, so a retained cell names bytes that exist.

        Raises:
            CandidateNotRetainableError: If the provider was given nowhere to keep them.
        """
        if self._store is None:
            raise CandidateNotRetainableError(
                "this provider was asked to keep what it fitted and was given no store"
            )
        return self._store.put(repr(tuple(self._stated.of(cell))).encode())
