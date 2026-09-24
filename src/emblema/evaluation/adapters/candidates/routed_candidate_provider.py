from collections.abc import Mapping

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.exceptions import (
    InvalidCandidateVariantError,
    UnknownCandidateError,
)
from emblema.evaluation.domain.tuning.candidate_variant import CandidateVariant
from emblema.evaluation.ports.candidate_provider import CandidateProvider


class RoutedCandidateProvider:
    """Every kind of competitor behind one port, each name routed to whoever supplies it.

    A campaign that compares a network with a fit of trees is asked of one provider, because
    knowing which of them answers a given name would make the campaign depend on what its
    candidates are made of. What the campaign gets instead is a map from name to supplier, built
    where the process is assembled and fixed before anything runs.

    Routed by that map and never by trying one supplier and catching its refusal: a provider
    that refuses a name it does own — weights it does not serve, a task it cannot read — would
    then be indistinguishable from one that was simply asked the wrong question, and which
    answer a campaign got would depend on the order the suppliers happened to be tried in.
    """

    def __init__(self, by_candidate: Mapping[CandidateRef, CandidateProvider]) -> None:
        self._by_candidate = dict(by_candidate)

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        return self._supplier(candidate).describe(candidate)

    def evaluate(self, request: CandidateEvaluation) -> CellResult:
        return self._supplier(request.cell.candidate).evaluate(request)

    def _supplier(self, candidate: CandidateRef) -> CandidateProvider:
        """Whoever this process was told supplies ``candidate``.

        A variant is supplied by whoever supplies the candidate it varies.

        Raises:
            UnknownCandidateError: If nothing in this process supplies that name.
        """
        try:
            base = CandidateVariant.parse(candidate).base
        except InvalidCandidateVariantError as error:
            raise UnknownCandidateError(f"{candidate} names no variant: {error}") from error
        if base not in self._by_candidate:
            raise UnknownCandidateError(
                f"this process supplies no candidate {base}; it supplies "
                f"{sorted(str(ref) for ref in self._by_candidate)}"
            )
        return self._by_candidate[base]
