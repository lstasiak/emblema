from dataclasses import dataclass, replace
from math import sqrt
from typing import Self

from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef, TaskId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.campaign_design import CampaignDesign
from emblema.evaluation.domain.campaign.campaign_verdict import CampaignVerdict
from emblema.evaluation.domain.campaign.candidate_comparison import CandidateComparison
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.exceptions import (
    CampaignCellAlreadyRecordedError,
    CampaignClosedError,
    CampaignNotCompletedError,
    IncompleteCampaignError,
    InvalidCampaignDesignError,
    SelectionHasNoVerdictError,
    SelectionNotReadableError,
    UnknownCampaignCellError,
    UnknownCandidateError,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.statistics.comparison_verdict import ComparisonVerdict
from emblema.evaluation.domain.statistics.error_over_repeats import ErrorOverRepeats
from emblema.evaluation.domain.statistics.paired_difference import PairedDifference
from emblema.evaluation.domain.statistics.paired_unit_errors import PairedUnitErrors
from emblema.evaluation.domain.statistics.practical_floor import PracticalFloor
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.domain.tuning.candidate_variant import CandidateVariant
from emblema.evaluation.domain.tuning.one_standard_error_rule import OneStandardErrorRule
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.compute import ComputeTier
from emblema.shared.kernel.timestamps import UtcDateTime


@dataclass(frozen=True, kw_only=True)
class _Measured:
    """A pairing measured and not yet judged: what a verdict is read from, rule by rule."""

    candidate: CandidateRef
    budget: LabelBudget
    control_error: ErrorOverRepeats
    candidate_error: ErrorOverRepeats
    difference: PairedDifference
    floor: PracticalFloor

    def judged(self, verdict: ComparisonVerdict) -> CandidateComparison:
        return CandidateComparison(
            candidate=self.candidate,
            budget=self.budget,
            control_error=self.control_error,
            candidate_error=self.candidate_error,
            difference=self.difference,
            floor=self.floor,
            verdict=verdict,
        )


@dataclass(frozen=True, kw_only=True)
class EvaluationCampaign:
    """A designed comparison and the record of running it, which together make a verdict possible.

    The campaign holds a result per cell and finishes only once every cell of its grid has one.
    That is the invariant the whole thing exists for: a grid read while part of it is missing
    reports the cells that happened to finish, and which those are is never independent of what
    they found — a diverging arm is exactly the one that takes longest or crashes. So a verdict
    is refused until the grid is whole, and a campaign that has finished refuses further cells.

    A dropped process loses nothing: what has been recorded is stored, and what is left is the
    grid minus it. Resuming is asking the campaign what is still pending.

    A campaign whose purpose is selection chooses among variants of a candidate rather than
    comparing candidates: it is scored inside the tuning side, and what it answers is which
    variant a comparison should run at each budget.

    Invariants: every result names a cell of the design's grid; no cell is recorded twice; a
    campaign is finished only with no cell pending; a selection campaign divides the tuning
    side and no other campaign does.

    Attributes:
        campaign_id: Identity of the campaign.
        task: Task every candidate answers.
        purpose: What the campaign is for, which decides whether its runs may see the task's
            frozen side.
        tier: Hardware class every one of its runs is measured on.
        design: Who competes, over what, and how the results will be read.
        results: What has been run so far, in the order it was recorded.
        opened_at: When the campaign was designed.
        completed_at: When its grid was finished; ``None`` while any cell is pending.
    """

    campaign_id: CampaignId
    task: TaskId
    purpose: RunPurpose
    tier: ComputeTier
    design: CampaignDesign
    results: tuple[CellResult, ...]
    opened_at: UtcDateTime
    completed_at: UtcDateTime | None

    def __post_init__(self) -> None:
        planned = set(self.design.cells())
        recorded = [result.cell for result in self.results]
        for cell in recorded:
            if cell not in planned:
                raise UnknownCampaignCellError(f"{cell} is not a cell of this campaign's grid")
        if len(set(recorded)) != len(recorded):
            raise CampaignCellAlreadyRecordedError("a cell of the grid is recorded twice")
        if self.completed_at is not None and len(recorded) != len(planned):
            raise IncompleteCampaignError(
                f"a campaign finished with {len(planned) - len(recorded)} cells still pending"
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
            results=(),
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
        return len(self.results) + (1 if self.is_finished else 0)

    @property
    def is_complete(self) -> bool:
        """Whether every cell of the grid has run."""
        return not self.pending()

    @property
    def is_finished(self) -> bool:
        """Whether the campaign has been closed and its verdict may be asked for."""
        return self.completed_at is not None

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
        recorded = {result.cell for result in self.results}
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

    def selected(self, candidate: CandidateRef, budget: LabelBudget) -> CandidateRef:
        """The variant of ``candidate`` this selection chooses at ``budget``, by its rule.

        The variants are the candidates of the grid whose name is ``candidate`` with knobs
        turned, the base included; each is read by its error in every repeat, and the rule of
        one standard error chooses among them.

        Raises:
            SelectionNotReadableError: If this is not a finished selection, or it holds no
                variant of that candidate at that budget besides the base.
        """
        holdout = self.design.inner_holdout
        if self.purpose is not RunPurpose.SELECTION or holdout is None or not self.is_finished:
            raise SelectionNotReadableError(
                f"campaign {self.campaign_id} is not a finished selection"
            )
        variants = [
            named
            for named in self.design.candidates
            if CandidateVariant.parse(named.ref).base == candidate
        ]
        defaults = [named for named in variants if named.ref == candidate]
        if not defaults or len(variants) < 2 or budget not in self.design.budgets:
            raise SelectionNotReadableError(
                f"campaign {self.campaign_id} holds no choice between {candidate} and a variant "
                f"of it at {budget}"
            )
        default = defaults[0].method
        return OneStandardErrorRule().choose(
            errors={
                named.ref: [result.rmse for result in self.results_of(named.ref, budget)]
                for named in variants
            },
            closeness={named.ref: named.method.departure_from(default) for named in variants},
            test_to_train=holdout.test_to_train,
        )

    def record(self, result: CellResult) -> Self:
        """The campaign with one more cell run.

        Raises:
            CampaignClosedError: If the campaign has already finished.
            UnknownCampaignCellError: If the result names a cell the grid does not hold.
            CampaignCellAlreadyRecordedError: If that cell has already been recorded.
        """
        if self.is_finished:
            raise CampaignClosedError(
                f"campaign {self.campaign_id} has finished and records no further cell"
            )
        return replace(self, results=(*self.results, result))

    def complete(self, at: UtcDateTime) -> Self:
        """The campaign closed, which is what makes its verdict readable.

        Raises:
            IncompleteCampaignError: If any cell of the grid is still pending.
            CampaignClosedError: If the campaign has already finished.
        """
        if self.is_finished:
            raise CampaignClosedError(f"campaign {self.campaign_id} has already finished")
        return replace(self, completed_at=at)

    @property
    def selects(self) -> bool:
        """Whether this campaign chooses among variants rather than compares candidates."""
        return self.purpose is RunPurpose.SELECTION

    def verdict(self) -> CampaignVerdict:
        """What the campaign concluded, read by the rules its design registered.

        Raises:
            CampaignNotCompletedError: If the campaign has not finished.
            SelectionHasNoVerdictError: If the campaign is a selection, whose repeats score
                different units and which is asked what it chose instead.
            InvalidPairedUnitErrorsError: If the candidate and the control were scored on
                different units.
        """
        if self.selects:
            raise SelectionHasNoVerdictError(
                f"campaign {self.campaign_id} is a selection: it is asked which variant it "
                "chose, not what it concluded"
            )
        if not self.is_finished:
            raise CampaignNotCompletedError(
                f"campaign {self.campaign_id} has no verdict until it has finished"
            )
        rules = self.design.rules
        endpoint = self._measured(self.design.endpoint, self.design.endpoint_budget)
        pairings = tuple(
            (candidate.ref, budget)
            for candidate in self.design.contenders
            for budget in self.design.budgets
            if not self.design.is_endpoint(candidate.ref, budget)
        )
        measured = tuple(self._measured(candidate, budget) for candidate, budget in pairings)
        rejections = rules.secondary_rejections([cell.difference.p_value for cell in measured])
        return CampaignVerdict(
            control=self.design.control,
            read_on=self.purpose,
            endpoint=endpoint.judged(rules.endpoint_verdict(endpoint.difference, endpoint.floor)),
            secondary=tuple(
                cell.judged(rules.secondary_verdict(cell.difference, cell.floor, rejected=rejected))
                for cell, rejected in zip(measured, rejections, strict=True)
            ),
        )

    def results_of(self, candidate: CandidateRef, budget: LabelBudget) -> tuple[CellResult, ...]:
        """Every repeat of one cell's pairing, in seed order."""
        return tuple(
            sorted(
                (
                    result
                    for result in self.results
                    if result.cell.candidate == candidate and result.cell.budget == budget
                ),
                key=lambda result: result.cell.seed,
            )
        )

    def rmse_of(self, candidate: CandidateRef, budget: LabelBudget) -> float:
        """What the candidate scored at one budget, pooled over its repeats.

        Pooled rather than averaged: the squared errors and the window counts of every repeat
        are added before the root is taken, so a repeat that answered more windows weighs more,
        which is what pairing a cell against the control already does.

        Raises:
            UnknownCandidateError: If no repeat of that pairing has run.
        """
        results = self.results_of(candidate, budget)
        if not results:
            raise UnknownCandidateError(
                f"no run of {candidate} at a budget of {budget.text()} to score"
            )
        squared = sum(error.squared_error for result in results for error in result.errors)
        windows = sum(error.windows for result in results for error in result.errors)
        return sqrt(squared / windows)

    def artifact_of(self, candidate: CandidateRef) -> ArtifactRef | None:
        """The candidate as the campaign kept it, or ``None`` where it kept none."""
        for result in self.results:
            if result.cell.candidate == candidate and self.design.retains(result.cell):
                return result.artifact
        return None

    def _measured(self, candidate: CandidateRef, budget: LabelBudget) -> _Measured:
        """The candidate against the control at one budget, before a rule is applied to it."""
        control = self.results_of(self.design.control, budget)
        contender = self.results_of(candidate, budget)
        paired = PairedUnitErrors.pooled(
            [result.errors for result in control], [result.errors for result in contender]
        )
        control_error = ErrorOverRepeats.of(paired.rmse_control, [r.rmse for r in control])
        return _Measured(
            candidate=candidate,
            budget=budget,
            control_error=control_error,
            candidate_error=ErrorOverRepeats.of(paired.rmse_candidate, [r.rmse for r in contender]),
            difference=self.design.bootstrap.compare(paired),
            floor=self.design.rules.floor_of(control_error),
        )
