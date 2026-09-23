from enum import StrEnum


class CandidateStanding(StrEnum):
    """Where a candidate ended up against the campaign's control arm, in published words.

    Narrower on purpose than the verdict this context reasons with. The full taxonomy is written
    against a registration — it distinguishes a reduction that missed the share promised
    beforehand from one that cleared it — and means nothing to a reader who has not read that
    registration. What travels is what another context can act on: whether the advantage was
    established, at the operating point the campaign was designed around.

    Attributes:
        CONTROL: The arm the others were measured against, which has no standing of its own.
        ESTABLISHED: The advantage over the control held under the campaign's own rules.
        NOT_ESTABLISHED: The advantage was not shown; the candidate may still be the better
            choice on other grounds, and this says only that this campaign did not show it.
        WORSE: The candidate's error was higher than the control's by more than the campaign
            treats as noise.
    """

    CONTROL = "control"
    ESTABLISHED = "established"
    NOT_ESTABLISHED = "not_established"
    WORSE = "worse"
