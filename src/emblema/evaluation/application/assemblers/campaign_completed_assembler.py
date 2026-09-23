from emblema.evaluation.contracts.candidate_metric import CandidateMetric
from emblema.evaluation.contracts.candidate_standing import CandidateStanding
from emblema.evaluation.contracts.evaluated_candidate import EvaluatedCandidate
from emblema.evaluation.contracts.events import CampaignCompleted
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.campaign_verdict import CampaignVerdict
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.statistics.comparison_verdict import ComparisonVerdict
from emblema.shared.events.domain_event import EventId
from emblema.shared.kernel.timestamps import UtcDateTime

ERROR = "rmse"

# What the context's own taxonomy amounts to for a reader who has not seen the registration a
# campaign was written against. A reduction that is real but smaller than the share promised
# beforehand still establishes an advantage; one the practical floor swallows does not, whatever
# its interval says.
_STANDINGS = {
    ComparisonVerdict.CONFIRMED: CandidateStanding.ESTABLISHED,
    ComparisonVerdict.DISTINGUISHABLE: CandidateStanding.ESTABLISHED,
    ComparisonVerdict.BELOW_REGISTERED_REDUCTION: CandidateStanding.ESTABLISHED,
    ComparisonVerdict.PRACTICALLY_NIL: CandidateStanding.NOT_ESTABLISHED,
    ComparisonVerdict.INDISTINGUISHABLE: CandidateStanding.NOT_ESTABLISHED,
    ComparisonVerdict.WORSE: CandidateStanding.WORSE,
}


class CampaignCompletedAssembler:
    """Builds the one message a finished campaign publishes, out of this context's own model.

    This is the Evaluation context's side of its published language, not an anti-corruption
    layer: it translates outwards. Two things are narrowed on the way. A candidate's standing
    comes out in words that mean something without the campaign's registration in hand, since a
    consumer cannot read a verdict written against a document it has never seen. And every cell
    of the grid is dropped in favour of one figure per budget, for the reason the message itself
    states.
    """

    def assemble(
        self,
        campaign: EvaluationCampaign,
        verdict: CampaignVerdict,
        *,
        event_id: EventId,
        occurred_at: UtcDateTime,
    ) -> CampaignCompleted:
        """The finished campaign as the message other contexts read it by."""
        return CampaignCompleted(
            event_id=event_id,
            occurred_at=occurred_at,
            campaign=campaign.campaign_id,
            task=campaign.task,
            verdict=verdict.sentence(),
            candidates=tuple(
                self._candidate(campaign, verdict, candidate)
                for candidate in campaign.design.candidates
            ),
        )

    def _candidate(
        self,
        campaign: EvaluationCampaign,
        verdict: CampaignVerdict,
        candidate: CampaignCandidate,
    ) -> EvaluatedCandidate:
        return EvaluatedCandidate(
            candidate=candidate.ref,
            kind=candidate.kind,
            artifact=campaign.artifact_of(candidate.ref),
            standing=self._standing(campaign, verdict, candidate.ref),
            metrics=tuple(
                CandidateMetric(
                    metric=ERROR,
                    budget=budget.windows,
                    value=campaign.rmse_of(candidate.ref, budget),
                    repeats=len(campaign.results_of(candidate.ref, budget)),
                )
                for budget in campaign.design.budgets
            ),
        )

    @staticmethod
    def _standing(
        campaign: EvaluationCampaign, verdict: CampaignVerdict, candidate: CandidateRef
    ) -> CandidateStanding:
        """Where a candidate stands at the budget the campaign's claim was made at."""
        if candidate == campaign.design.control:
            return CandidateStanding.CONTROL
        comparison = verdict.get_comparison(candidate, campaign.design.endpoint_budget)
        return _STANDINGS[comparison.verdict]
