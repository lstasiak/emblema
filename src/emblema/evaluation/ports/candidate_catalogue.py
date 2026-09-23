from typing import Protocol

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate


class CandidateCatalogue(Protocol):
    """Says what a candidate is, in the terms a design is stated in.

    Declaring a comparison and running one are two things, and only the first is asked of this.
    A campaign's design has to record what each competitor was set to, and a figure written down
    beside a candidate can disagree with what the candidate does — so the design asks whoever
    supplies it. Nothing about that answer needs the ability to produce it: naming the arms of a
    backbone needs the arms and the schedule, naming the baselines needs the baselines and the
    knobs they fit by, and neither needs a runtime.

    That is what keeps the process which declares a campaign from carrying every stack the grid
    will be run with. On the platform this is developed on it is not merely wasteful: the
    training stack and the one the baselines are fitted with cannot share a process at all.
    """

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        """What this candidate is, in the terms a design is stated in.

        Raises:
            UnknownCandidateError: If this catalogue holds no such candidate.
        """
        ...
