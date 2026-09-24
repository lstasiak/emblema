from collections.abc import Mapping

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.exceptions import UnknownCandidateError
from emblema.evaluation.ports.candidate_catalogue import CandidateCatalogue


class RoutedCandidateCatalogue:
    """Every kind of competitor behind one catalogue, each name routed to whoever holds it.

    The describing half of ``RoutedCandidateProvider``, for the process that declares a campaign
    and runs nothing: the same map from name to holder, for the same reason it is a map and not
    a chain of holders tried in turn.
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
