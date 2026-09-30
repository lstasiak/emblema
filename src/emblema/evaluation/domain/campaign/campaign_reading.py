from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import sqrt
from statistics import fmean
from typing import Self

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.campaign_verdict import CampaignVerdict
from emblema.evaluation.domain.campaign.candidate_comparison import CandidateComparison
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.exceptions import (
    CampaignNotCompletedError,
    SelectionHasNoVerdictError,
    SelectionNotReadableError,
    UnknownCampaignCellError,
    UnknownCandidateError,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from emblema.evaluation.domain.statistics.comparison_verdict import ComparisonVerdict
from emblema.evaluation.domain.statistics.error_over_repeats import ErrorOverRepeats
from emblema.evaluation.domain.statistics.paired_difference import PairedDifference
from emblema.evaluation.domain.statistics.paired_unit_errors import PairedUnitErrors
from emblema.evaluation.domain.statistics.paired_unit_rankings import PairedUnitRankings
from emblema.evaluation.domain.statistics.paired_units import PairedUnits
from emblema.evaluation.domain.statistics.practical_floor import PracticalFloor
from emblema.evaluation.domain.tuning.one_standard_error_rule import OneStandardErrorRule
from emblema.shared.kernel.artifacts import ArtifactRef
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
class CampaignReading:
    """A campaign with the results of every cell it recorded: what its conclusion is read from.

    The campaign holds which cells have run; the results, thousands of answers a cell, are
    loaded here once by whoever reads the verdict, the choice of a selection or the figures,
    and by nothing that orders, runs or records a cell.

    Invariants: the results are exactly one per recorded cell of the campaign.

    Attributes:
        campaign: The campaign as it stands.
        results: What each recorded cell produced.
    """

    campaign: EvaluationCampaign
    results: tuple[CellResult, ...]

    def __post_init__(self) -> None:
        self.campaign.accept_results([result.cell for result in self.results])

    def record(self, result: CellResult) -> Self:
        """The reading with one more cell run and what it produced.

        Raises:
            CampaignClosedError: If the campaign has already finished.
            UnknownCampaignCellError: If the grid does not hold the result's cell.
            CampaignCellAlreadyRecordedError: If that cell has already been recorded.
        """
        return replace(
            self, campaign=self.campaign.record(result.cell), results=(*self.results, result)
        )

    def complete(self, at: UtcDateTime) -> Self:
        """The reading of the campaign closed at ``at``.

        Raises:
            IncompleteCampaignError: If any cell of the grid is still pending.
            CampaignClosedError: If the campaign has already finished.
        """
        return replace(self, campaign=self.campaign.complete(at))

    def get_result(self, cell: CampaignCell) -> CellResult:
        """What ``cell`` produced.

        Raises:
            UnknownCampaignCellError: If the campaign has not recorded that cell.
        """
        for result in self.results:
            if result.cell == cell:
                return result
        raise UnknownCampaignCellError(f"campaign {self.campaign.campaign_id} has not run {cell}")

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

    def error_of(self, candidate: CandidateRef, budget: LabelBudget) -> float:
        """What the candidate scored at one budget under the design's measure, over its repeats.

        Pooled as a comparison pools the side: squared errors and window counts added over the
        repeats before the root is taken, so a repeat that answered more windows weighs more; an
        area under the ROC curve averaged over the repeats, each its own model.

        Raises:
            UnknownCandidateError: If no repeat of that pairing has run.
        """
        results = self._ran(candidate, budget)
        match self.campaign.design.measure:
            case ErrorMeasure.RMSE:
                return sqrt(self.mean_squared_error_of(candidate, budget))
            case ErrorMeasure.AUROC_SHORTFALL:
                return fmean(result.error_under(ErrorMeasure.AUROC_SHORTFALL) for result in results)

    def mean_squared_error_of(self, candidate: CandidateRef, budget: LabelBudget) -> float:
        """The mean squared error of the candidate's answers at one budget, over its repeats.

        Every repeat's squared errors and window counts added before the mean is taken. For a
        binary task the answers are probabilities and this is the Brier score: how far they lie
        from the outcomes, which a ranking measure does not see.

        Raises:
            UnknownCandidateError: If no repeat of that pairing has run.
        """
        results = self._ran(candidate, budget)
        squared = sum(error.squared_error for result in results for error in result.errors)
        windows = sum(error.windows for result in results for error in result.errors)
        return squared / windows

    def artifact_of(self, candidate: CandidateRef) -> ArtifactRef | None:
        """The candidate as the campaign kept it, or ``None`` where it kept none."""
        for result in self.results:
            if result.cell.candidate == candidate and self.campaign.design.retains(result.cell):
                return result.artifact
        return None

    def selected(self, candidate: CandidateRef, budget: LabelBudget) -> CandidateRef:
        """The variant of ``candidate`` this selection chooses at ``budget``, by its rule.

        The variants are the candidates of the grid whose name is ``candidate`` with knobs
        turned, the setting they are turned around included; each is read by its error in every
        repeat, and the rule of one standard error chooses among them, ties broken towards that
        setting.

        Raises:
            SelectionNotReadableError: If this is not a finished selection, it holds fewer than
                two variants of that candidate at that budget, no one variant is the setting
                the others are turned around, or every repeat held the same units out.
        """
        campaign = self.campaign
        holdout = campaign.design.inner_holdout
        variants = campaign.variants_of(candidate)
        if holdout is None or budget not in campaign.design.budgets:
            raise SelectionNotReadableError(
                f"campaign {campaign.campaign_id} holds no choice between {candidate} and a "
                f"variant of it at {budget}"
            )
        if holdout.division_seed is not None:
            raise SelectionNotReadableError(
                f"campaign {campaign.campaign_id} holds the same units out in every repeat, "
                f"which the rule's correction does not describe"
            )
        default = campaign.turned_around(candidate).method
        return OneStandardErrorRule().choose(
            errors={
                named.ref: [
                    result.error_under(campaign.design.measure)
                    for result in self.results_of(named.ref, budget)
                ]
                for named in variants
            },
            closeness={named.ref: named.method.departure_from(default) for named in variants},
            test_to_train=holdout.test_to_train,
        )

    def verdict(self) -> CampaignVerdict:
        """What the campaign concluded, read by the rules its design registered.

        Raises:
            CampaignNotCompletedError: If the campaign has not finished.
            SelectionHasNoVerdictError: If the campaign is a selection, whose repeats score
                different units and which is asked what it chose instead.
            InvalidPairedUnitErrorsError: If the candidate and the control were scored on
                different units.
        """
        campaign = self.campaign
        if campaign.selects:
            raise SelectionHasNoVerdictError(
                f"campaign {campaign.campaign_id} is a selection: it is asked which variant it "
                "chose, not what it concluded"
            )
        if not campaign.is_finished:
            raise CampaignNotCompletedError(
                f"campaign {campaign.campaign_id} has no verdict until it has finished"
            )
        design = campaign.design
        rules = design.rules
        endpoint = self._measured(design.endpoint, design.endpoint_budget)
        pairings = tuple(
            (candidate.ref, budget)
            for candidate in design.contenders
            for budget in design.budgets
            if not design.is_endpoint(candidate.ref, budget)
        )
        measured = tuple(self._measured(candidate, budget) for candidate, budget in pairings)
        rejections = rules.secondary_rejections([cell.difference.p_value for cell in measured])
        return CampaignVerdict(
            control=design.control,
            read_on=campaign.purpose,
            measure=design.measure,
            endpoint=endpoint.judged(rules.endpoint_verdict(endpoint.difference, endpoint.floor)),
            secondary=tuple(
                cell.judged(rules.secondary_verdict(cell.difference, cell.floor, rejected=rejected))
                for cell, rejected in zip(measured, rejections, strict=True)
            ),
        )

    def _ran(self, candidate: CandidateRef, budget: LabelBudget) -> tuple[CellResult, ...]:
        results = self.results_of(candidate, budget)
        if not results:
            raise UnknownCandidateError(
                f"no run of {candidate} at a budget of {budget.text()} to score"
            )
        return results

    def _measured(self, candidate: CandidateRef, budget: LabelBudget) -> _Measured:
        """The candidate against the control at one budget, before a rule is applied to it."""
        design = self.campaign.design
        control = self.results_of(design.control, budget)
        contender = self.results_of(candidate, budget)
        paired = self._paired(control, contender)
        control_error = ErrorOverRepeats.of(
            paired.error_control, [r.error_under(design.measure) for r in control]
        )
        return _Measured(
            candidate=candidate,
            budget=budget,
            control_error=control_error,
            candidate_error=ErrorOverRepeats.of(
                paired.error_candidate, [r.error_under(design.measure) for r in contender]
            ),
            difference=design.bootstrap.compare(paired),
            floor=design.rules.floor_of(control_error),
        )

    def _paired(
        self, control: Sequence[CellResult], contender: Sequence[CellResult]
    ) -> PairedUnits:
        """The repeats of both sides paired unit by unit, in the form the design's measure reads."""
        match self.campaign.design.measure:
            case ErrorMeasure.RMSE:
                return PairedUnitErrors.pooled(
                    [result.errors for result in control], [result.errors for result in contender]
                )
            case ErrorMeasure.AUROC_SHORTFALL:
                return PairedUnitRankings(
                    control=tuple(result.ranking() for result in control),
                    candidate=tuple(result.ranking() for result in contender),
                )
