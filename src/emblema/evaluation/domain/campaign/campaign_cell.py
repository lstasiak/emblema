from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.labels.label_budget import LabelBudget


@dataclass(frozen=True, kw_only=True)
class CampaignCell:
    """One point of a campaign's grid: a candidate at a budget of labels under a seed.

    The coordinates and nothing else, so a cell is a key: what a campaign has still to run, what
    a result is recorded against, and what a scheduler is handed are all the same value.

    Attributes:
        candidate: Which competitor this point measures.
        budget: How many labelled windows it learns from here.
        seed: Which repeat of that pairing this is; seeds are repeats of one cell, not a
            dimension anything is averaged over before the units are.
    """

    candidate: CandidateRef
    budget: LabelBudget
    seed: int

    def __str__(self) -> str:
        return f"{self.candidate} at {self.budget.text()} under seed {self.seed}"
