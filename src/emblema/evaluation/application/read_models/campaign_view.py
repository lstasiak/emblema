from dataclasses import dataclass

from emblema.evaluation.application.read_models.campaign_summary import CampaignSummary
from emblema.evaluation.application.read_models.candidate_curve import CandidateCurve
from emblema.evaluation.application.read_models.verdict_view import VerdictView


@dataclass(frozen=True, kw_only=True)
class CampaignView:
    """One campaign in full: what a list shows of it, every candidate's curve, and the verdict.

    The curves are readable while the grid is still running, budget by budget as repeats come
    in; the verdict only once the grid is whole, and never for a selection, which chooses a
    variant and concludes nothing.

    Attributes:
        summary: What the campaign compares, on what, and how far its grid got.
        curves: One curve per candidate, in the order the campaign reports them.
        verdict: What the campaign concluded; ``None`` while it runs, or for a selection.
    """

    summary: CampaignSummary
    curves: tuple[CandidateCurve, ...]
    verdict: VerdictView | None
