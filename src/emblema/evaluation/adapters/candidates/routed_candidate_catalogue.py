from collections.abc import Mapping

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.exceptions import (
    InvalidCandidateVariantError,
    UnknownCandidateError,
)
from emblema.evaluation.domain.tuning.candidate_variant import CandidateVariant
from emblema.evaluation.ports.candidate_catalogue import CandidateCatalogue


class RoutedCandidateCatalogue:
    """Every kind of competitor behind one catalogue, each name routed to whoever holds it.

    The describing half of ``RoutedCandidateProvider``, for the process that declares a campaign
    and runs nothing: the same map from name to holder, for the same reason it is a map and not
    a chain of holders tried in turn. A variant is routed by the candidate it varies, so its
    holder is the one that holds the base and knows the knobs.
    """

    def __init__(self, by_candidate: Mapping[CandidateRef, CandidateCatalogue]) -> None:
        self._by_candidate = dict(by_candidate)

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        return self._holder(candidate).describe(candidate)

    def _holder(self, candidate: CandidateRef) -> CandidateCatalogue:
        """Whoever this process was told holds ``candidate``, or the candidate it varies.

        Raises:
            UnknownCandidateError: If nothing in this process holds that name.
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
