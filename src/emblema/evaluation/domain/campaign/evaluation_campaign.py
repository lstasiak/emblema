from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Self

from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef, TaskId
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.campaign_design import CampaignDesign
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.exceptions import (
    CampaignCellAlreadyRecordedError,
    CampaignClosedError,
    CampaignNotCompletedError,
    IncompleteCampaignError,
    InvalidCampaignDesignError,
    SelectionNotReadableError,
    UnknownCampaignCellError,
)
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.domain.tuning.candidate_variant import CandidateVariant
from emblema.shared.kernel.compute import ComputeTier
from emblema.shared.kernel.timestamps import UtcDateTime


@dataclass(frozen=True, kw_only=True)
class EvaluationCampaign:
    """A designed comparison and the record of which of its cells have run.

    The campaign finishes only once every cell of its grid has run. That is the invariant the
    whole thing exists for: a grid read while part of it is missing reports the cells that
    happened to finish, and which those are is never independent of what they found — a
    diverging arm is exactly the one that takes longest or crashes. So the campaign refuses to
    close while a cell is pending, and once closed refuses further cells.

    The campaign records which cells have run, not what they produced: its invariants are about
    the grid, and the processes that order, run and record cells need nothing more. What the
    cells produced is read beside it, as a ``CampaignReading``, by whoever reads the verdict,
    the selection or the figures.

    A dropped process loses nothing: what has been recorded is stored, and what is left is the
    grid minus it. Resuming is asking the campaign what is still pending.

    A campaign whose purpose is selection chooses among variants of a candidate rather than
    comparing candidates: it is scored inside the tuning side, and what it answers is which
    variant a comparison should run at each budget.

    Invariants: every recorded cell is a cell of the design's grid; no cell is recorded twice; a
    campaign is finished only with no cell pending; a selection campaign divides the tuning
    side and no other campaign does.

    Attributes:
        campaign_id: Identity of the campaign.
        task: Task every candidate answers.
        purpose: What the campaign is for, which decides whether its runs may see the task's
            frozen side.
        tier: Hardware class every one of its runs is measured on.
        design: Who competes, over what, and how the results will be read.
        recorded: The cells that have run, in the order they were recorded.
        opened_at: When the campaign was designed.
        completed_at: When its grid was finished; ``None`` while any cell is pending.
    """

    campaign_id: CampaignId
    task: TaskId
    purpose: RunPurpose
    tier: ComputeTier
    design: CampaignDesign
    recorded: tuple[CampaignCell, ...]
    opened_at: UtcDateTime
    completed_at: UtcDateTime | None

    def __post_init__(self) -> None:
        planned = set(self.design.cells())
        for cell in self.recorded:
            if cell not in planned:
                raise UnknownCampaignCellError(f"{cell} is not a cell of this campaign's grid")
        if len(set(self.recorded)) != len(self.recorded):
            raise CampaignCellAlreadyRecordedError("a cell of the grid is recorded twice")
        if self.completed_at is not None and len(self.recorded) != len(planned):
            raise IncompleteCampaignError(
                f"a campaign finished with {len(planned) - len(self.recorded)} cells still pending"
            )
        if (self.purpose is RunPurpose.SELECTION) != (self.design.inner_holdout is not None):
            raise InvalidCampaignDesignError(
                "a selection campaign divides the tuning side, and only a selection campaign does"
            )

    @classmethod
    def designed(
        cls,
        *,
        campaign_id: CampaignId,
        task: TaskId,
        purpose: RunPurpose,
        tier: ComputeTier,
        design: CampaignDesign,
        opened_at: UtcDateTime,
    ) -> Self:
        """A campaign with its grid laid out and nothing run yet."""
        return cls(
            campaign_id=campaign_id,
            task=task,
            purpose=purpose,
            tier=tier,
            design=design,
            recorded=(),
            opened_at=opened_at,
            completed_at=None,
        )

    @property
    def revision(self) -> int:
        """How many times this campaign has changed since it was designed.

        A campaign gains cells and then closes, and nothing else about it ever moves: who
        competes, over what and how the result will be read are settled before anything runs.
        So counting the cells and the closing counts the changes, and the number rises by one
        with every one of them without a field to keep in step with the state. It is what a
        writer claims to have read when it writes the campaign back.
        """
        return len(self.recorded) + (1 if self.is_finished else 0)

    @property
    def cells_recorded(self) -> int:
        return len(self.recorded)

    @property
    def is_complete(self) -> bool:
        """Whether every cell of the grid has run."""
        return not self.pending()

    @property
    def is_finished(self) -> bool:
        """Whether the campaign has been closed and its verdict may be asked for."""
        return self.completed_at is not None

    @property
    def selects(self) -> bool:
        """Whether this campaign chooses among variants rather than compares candidates."""
        return self.purpose is RunPurpose.SELECTION

    def completion(self) -> UtcDateTime:
        """When the campaign was closed.

        Raises:
            CampaignNotCompletedError: If it has not been.
        """
        if self.completed_at is None:
            raise CampaignNotCompletedError(f"campaign {self.campaign_id} has not finished")
        return self.completed_at

    def pending(self) -> tuple[CampaignCell, ...]:
        """The cells still to run, in the grid's order — what resuming a campaign asks for."""
        recorded = set(self.recorded)
        return tuple(cell for cell in self.design.cells() if cell not in recorded)

    def evaluation_of(self, cell: CampaignCell) -> CandidateEvaluation:
        """What running ``cell`` asks of its candidate, whichever process runs it.

        One place builds the request, so a cell handed to a queue and a cell written into an
        order for another machine ask the candidate the same thing.

        Raises:
            UnknownCampaignCellError: If the grid does not hold that cell.
        """
        if cell not in self.design.cells():
            raise UnknownCampaignCellError(f"{cell} is not a cell of campaign {self.campaign_id}")
        return CandidateEvaluation(
            task=self.task,
            cell=cell,
            purpose=self.purpose,
            retain=self.design.retains(cell),
            declared=self.design.declared_for(cell),
            holdout=self.design.inner_holdout,
        )

    def record(self, cell: CampaignCell) -> Self:
        """The campaign with one more cell run.

        Raises:
            CampaignClosedError: If the campaign has already finished.
            UnknownCampaignCellError: If the grid does not hold that cell.
            CampaignCellAlreadyRecordedError: If that cell has already been recorded.
        """
        if self.is_finished:
            raise CampaignClosedError(
                f"campaign {self.campaign_id} has finished and records no further cell"
            )
        return replace(self, recorded=(*self.recorded, cell))

    def complete(self, at: UtcDateTime) -> Self:
        """The campaign closed, which is what makes its verdict readable.

        Raises:
            IncompleteCampaignError: If any cell of the grid is still pending.
            CampaignClosedError: If the campaign has already finished.
        """
        if self.is_finished:
            raise CampaignClosedError(f"campaign {self.campaign_id} has already finished")
        return replace(self, completed_at=at)

    def variants_of(self, candidate: CandidateRef) -> tuple[CampaignCandidate, ...]:
        """The candidates of the grid that are ``candidate`` with knobs turned, its setting too.

        Raises:
            SelectionNotReadableError: If this is not a finished selection, or holds fewer than
                two of them.
        """
        if self.purpose is not RunPurpose.SELECTION or not self.is_finished:
            raise SelectionNotReadableError(
                f"campaign {self.campaign_id} is not a finished selection"
            )
        variants = tuple(
            named
            for named in self.design.candidates
            if CandidateVariant.parse(named.ref).base == candidate
        )
        if len(variants) < 2:
            raise SelectionNotReadableError(
                f"campaign {self.campaign_id} holds no choice between {candidate} and a variant "
                "of it"
            )
        return variants

    def turned_around(self, candidate: CandidateRef) -> CampaignCandidate:
        """The setting this selection turns the knobs of ``candidate`` around, as it ran it.

        What a comparison tuned by this selection is held to beside the variant chosen: the
        candidate it names is the one whose knobs were turned, and this is how the selection
        described that candidate at its setting, a variant itself where the selection was
        declared around one.

        The candidate under its bare name when the selection holds it. A selection declared
        around a setting that is itself a variant — the arms under the tail of the window —
        holds no bare candidate; there it is the one variant the others depart from by the
        fewest knobs in all, which under knobs turned one at a time is the setting they were
        turned from.

        Raises:
            SelectionNotReadableError: If this is not a finished selection, holds fewer than
                two variants of that candidate, or no one variant is turned around.
        """
        variants = self.variants_of(candidate)
        for named in variants:
            if named.ref == candidate:
                return named
        knobs_away = {
            named.ref: sum(other.method.departure_from(named.method)[0] for other in variants)
            for named in variants
        }
        nearest, next_nearest = sorted(knobs_away.values())[:2]
        if nearest == next_nearest:
            raise SelectionNotReadableError(
                f"campaign {self.campaign_id} does not say which variant of {candidate} its "
                "knobs are turned around"
            )
        return next(named for named in variants if knobs_away[named.ref] == nearest)

    def accept_results(self, cells: Sequence[CampaignCell]) -> None:
        """Refuse results that are not exactly one per recorded cell.

        Raises:
            UnknownCampaignCellError: If a result names a cell the campaign has not recorded.
            CampaignCellAlreadyRecordedError: If a cell has two results.
            IncompleteCampaignError: If a recorded cell has no result.
        """
        recorded = set(self.recorded)
        for cell in cells:
            if cell not in recorded:
                raise UnknownCampaignCellError(
                    f"campaign {self.campaign_id} has not recorded {cell}"
                )
        if len(set(cells)) != len(cells):
            raise CampaignCellAlreadyRecordedError("a cell of the grid has two results")
        if len(cells) != len(recorded):
            raise IncompleteCampaignError(
                f"campaign {self.campaign_id} recorded {len(recorded) - len(cells)} cells "
                "whose results are missing"
            )
