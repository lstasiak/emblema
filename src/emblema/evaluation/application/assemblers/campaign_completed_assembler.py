from typing import ClassVar

from emblema.evaluation.contracts.candidate_metric import CandidateMetric
from emblema.evaluation.contracts.candidate_standing import CandidateStanding
from emblema.evaluation.contracts.evaluated_candidate import EvaluatedCandidate
from emblema.evaluation.contracts.events import CampaignCompleted
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.campaign_design import CampaignDesign
from emblema.evaluation.domain.campaign.campaign_reading import CampaignReading
from emblema.evaluation.domain.campaign.campaign_verdict import CampaignVerdict
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from emblema.evaluation.domain.statistics.comparison_verdict import ComparisonVerdict
from emblema.shared.events.domain_event import EventId
from emblema.shared.kernel.timestamps import UtcDateTime


class CampaignCompletedAssembler:
    """Builds the one message a finished campaign publishes, out of this context's own model.

    This is the Evaluation context's side of its published language, not an anti-corruption
    layer: it translates outwards. Two things are narrowed on the way. A candidate's standing
    comes out in words that mean something without the campaign's registration in hand, since a
    consumer cannot read a verdict written against a document it has never seen. And every cell
    of the grid is dropped in favour of one figure per budget, for the reason the message itself
    states. A campaign read by a ranking publishes the area under the ROC curve as it is usually
    stated, higher is better, and the Brier score beside it, since a consumer promoting a
    probability wants to know how far the probabilities are from the outcomes as well as how
    well they rank.

    Attributes:
        ERROR: The metric name a squared-error campaign publishes its figures under.
        AUROC: The metric name of the area a campaign read by ranking publishes.
        BRIER: The metric name of the Brier score published beside the area.
    """

    ERROR: ClassVar[str] = "rmse"
    AUROC: ClassVar[str] = "auroc"
    BRIER: ClassVar[str] = "brier"

    # What the context's own taxonomy amounts to for a reader who has not seen the registration
    # a campaign was written against. A reduction that is real but smaller than the share
    # promised beforehand still establishes an advantage; one the practical floor swallows does
    # not, whatever its interval says.
    _STANDINGS: ClassVar[dict[ComparisonVerdict, CandidateStanding]] = {
        ComparisonVerdict.CONFIRMED: CandidateStanding.ESTABLISHED,
        ComparisonVerdict.DISTINGUISHABLE: CandidateStanding.ESTABLISHED,
        ComparisonVerdict.BELOW_REGISTERED_REDUCTION: CandidateStanding.ESTABLISHED,
        ComparisonVerdict.PRACTICALLY_NIL: CandidateStanding.NOT_ESTABLISHED,
        ComparisonVerdict.INDISTINGUISHABLE: CandidateStanding.NOT_ESTABLISHED,
        ComparisonVerdict.WORSE: CandidateStanding.WORSE,
    }

    def assemble(
        self,
        reading: CampaignReading,
        verdict: CampaignVerdict,
        *,
        event_id: EventId,
        occurred_at: UtcDateTime,
    ) -> CampaignCompleted:
        """The finished campaign as the message other contexts read it by."""
        campaign = reading.campaign
        return CampaignCompleted(
            event_id=event_id,
            occurred_at=occurred_at,
            campaign=campaign.campaign_id,
            task=campaign.task,
            verdict=verdict.sentence(),
            candidates=tuple(
                self._candidate(reading, verdict, candidate)
                for candidate in campaign.design.candidates
            ),
        )

    def _candidate(
        self,
        reading: CampaignReading,
        verdict: CampaignVerdict,
        candidate: CampaignCandidate,
    ) -> EvaluatedCandidate:
        return EvaluatedCandidate(
            candidate=candidate.ref,
            kind=candidate.kind,
            artifact=reading.artifact_of(candidate.ref),
            standing=self._standing(reading.campaign.design, verdict, candidate.ref),
            metrics=tuple(
                CandidateMetric(
                    metric=metric,
                    budget=budget.windows,
                    value=value,
                    repeats=len(reading.results_of(candidate.ref, budget)),
                )
                for budget in reading.campaign.design.budgets
                for metric, value in self._figures(reading, candidate.ref, budget)
            ),
        )

    @classmethod
    def _figures(
        cls, reading: CampaignReading, candidate: CandidateRef, budget: LabelBudget
    ) -> tuple[tuple[str, float], ...]:
        """The figures a consumer reads a candidate by at one budget, named as it knows them."""
        match reading.campaign.design.measure:
            case ErrorMeasure.RMSE:
                return ((cls.ERROR, reading.error_of(candidate, budget)),)
            case ErrorMeasure.AUROC_SHORTFALL:
                return (
                    (cls.AUROC, 1.0 - reading.error_of(candidate, budget)),
                    (cls.BRIER, reading.mean_squared_error_of(candidate, budget)),
                )

    @classmethod
    def _standing(
        cls, design: CampaignDesign, verdict: CampaignVerdict, candidate: CandidateRef
    ) -> CandidateStanding:
        """Where a candidate stands at the budget the campaign's claim was made at."""
        if candidate == design.control:
            return CandidateStanding.CONTROL
        comparison = verdict.get_comparison(candidate, design.endpoint_budget)
        return cls._STANDINGS[comparison.verdict]
