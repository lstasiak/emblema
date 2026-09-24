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
    knobs they fit by, and neither needs a runtime — which is what lets the process that
    declares a campaign carry none of the stacks the grid is run with.
    """

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        """What this candidate is, in the terms a design is stated in.

        Raises:
            UnknownCandidateError: If this catalogue holds no such candidate.
        """
        ...
