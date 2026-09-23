from collections.abc import Mapping

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.exceptions import UnknownCandidateError
from emblema.evaluation.ports.candidate_catalogue import CandidateCatalogue


class RoutedCandidateCatalogue:
    """Every kind of competitor behind one catalogue, each name routed to whoever holds it.

    What a campaign is designed against is one catalogue, because knowing which of them answers
    a given name would make the design depend on what its candidates are made of. What it gets
    instead is a map from name to holder, built where the process is assembled.

    Routed by that map and never by trying one holder and catching its refusal: a holder that
    refuses a name it does own would then be indistinguishable from one asked the wrong
    question, and which answer a design got would depend on the order they were tried in.
    """

    def __init__(self, by_candidate: Mapping[CandidateRef, CandidateCatalogue]) -> None:
        self._by_candidate = dict(by_candidate)

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        if candidate not in self._by_candidate:
            raise UnknownCandidateError(
                f"this process supplies no candidate {candidate}; it supplies "
                f"{sorted(str(ref) for ref in self._by_candidate)}"
            )
        return self._by_candidate[candidate].describe(candidate)
