from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.exceptions import (
    InvalidCampaignDesignError,
    UnknownCandidateError,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.statistics.comparison_rules import ComparisonRules
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap


@dataclass(frozen=True, kw_only=True)
class CampaignDesign:
    """A campaign stated before anything runs: who competes, over what, and how it will be read.

    Everything a verdict depends on is fixed here, which is the reason the type exists. A grid
    whose candidates, budgets, seeds and rules were settled as results came in would let each
    choice be made knowing what it does to the answer, and no amount of arithmetic afterwards
    repairs that. The design is written down once and the campaign is then only the record of
    running it.

    One candidate is the control: every other is compared against it at each budget, on the same
    units under the same seeds. One of those comparisons is the endpoint — the single claim the
    campaign was designed to test — and the rest are the secondary family, held to a correction
    over the family the rules name rather than to their own intervals.

    Invariants: at least two candidates, none named twice; the control and the endpoint are
    among them and are not the same; at least one budget and one seed, none repeated; the
    endpoint's budget is one of the budgets; every candidate that shares the compute budget
    declares the same one; the secondary comparisons the grid holds fit the registered family.

    Attributes:
        candidates: Every competitor, the control included, in the order the campaign reports
            them.
        control: Which of them the others are measured against.
        endpoint: Which candidate the campaign's single claim is about.
        budgets: How many labelled windows each cell learns from, in reporting order.
        endpoint_budget: The budget the campaign's claim is made at.
        seeds: Repeats of every cell. The first is the reference seed: the run whose fitted
            candidate is kept, chosen before any of them has run so that keeping one is not a
            selection.
        rules: What the campaign's registration said a verdict requires.
        bootstrap: How the interval around each comparison is drawn.
    """

    candidates: tuple[CampaignCandidate, ...]
    control: CandidateRef
    endpoint: CandidateRef
    budgets: tuple[LabelBudget, ...]
    endpoint_budget: LabelBudget
    seeds: tuple[int, ...]
    rules: ComparisonRules
    bootstrap: PairedUnitBootstrap

    def __post_init__(self) -> None:
        self._check_candidates()
        self._check_axes()
        self._check_family()

    def _check_candidates(self) -> None:
        if len(self.candidates) < 2:
            raise InvalidCampaignDesignError(
                "a campaign compares a control with at least one other candidate"
            )
        refs = [candidate.ref for candidate in self.candidates]
        if len(set(refs)) != len(refs):
            raise InvalidCampaignDesignError(
                f"a candidate is named twice: {sorted(map(str, refs))}"
            )
        for role, ref in (("control", self.control), ("endpoint", self.endpoint)):
            if ref not in refs:
                raise InvalidCampaignDesignError(f"the {role} {ref} is not among the candidates")
        if self.control == self.endpoint:
            raise InvalidCampaignDesignError(
                f"the endpoint {self.endpoint} is the control it would be compared against"
            )
        budgets = {
            candidate.budget
            for candidate in self.candidates
            if candidate.kind.shares_the_compute_budget
        }
        if len(budgets) > 1:
            raise InvalidCampaignDesignError(
                "candidates that share the compute budget declare different ones: "
                + "; ".join(sorted(str(budget) for budget in budgets))
            )

    def _check_axes(self) -> None:
        for label, axis in (("budget", self.budgets), ("seed", self.seeds)):
            if not axis:
                raise InvalidCampaignDesignError(f"a campaign needs at least one {label}")
            if len(set(axis)) != len(axis):
                raise InvalidCampaignDesignError(f"a {label} is named twice")
        if self.endpoint_budget not in self.budgets:
            raise InvalidCampaignDesignError("the endpoint's budget is not one the campaign runs")

    def _check_family(self) -> None:
        secondary = len(self.contenders) * len(self.budgets) - 1
        if secondary > self.rules.secondary_family_size:
            raise InvalidCampaignDesignError(
                f"the grid holds {secondary} secondary comparisons, more than the "
                f"{self.rules.secondary_family_size} the registration names"
            )

    @property
    def contenders(self) -> tuple[CampaignCandidate, ...]:
        """Every candidate but the control, in reporting order."""
        return tuple(candidate for candidate in self.candidates if candidate.ref != self.control)

    @property
    def reference_seed(self) -> int:
        """The repeat whose fitted candidate the campaign keeps."""
        return self.seeds[0]

    def cells(self) -> tuple[CampaignCell, ...]:
        """Every point of the grid, in a fixed order: candidate, then budget, then seed."""
        return tuple(
            CampaignCell(candidate=candidate.ref, budget=budget, seed=seed)
            for candidate in self.candidates
            for budget in self.budgets
            for seed in self.seeds
        )

    def get_candidate(self, ref: CandidateRef) -> CampaignCandidate:
        """The candidate the design calls ``ref``.

        Raises:
            UnknownCandidateError: If the design names no such candidate.
        """
        for candidate in self.candidates:
            if candidate.ref == ref:
                return candidate
        raise UnknownCandidateError(f"the design names no candidate {ref}")

    def retains(self, cell: CampaignCell) -> bool:
        """Whether the candidate fitted in this cell is kept as an artifact.

        One cell per candidate: the operating point the campaign's claim is made at, under the
        reference seed. Keeping every cell would store a grid's worth of weights to promote one
        of them, and keeping the best would be choosing on the numbers the campaign produced.
        """
        return cell.budget == self.endpoint_budget and cell.seed == self.reference_seed

    def is_endpoint(self, candidate: CandidateRef, budget: LabelBudget) -> bool:
        """Whether this pairing is the single claim the campaign was designed to test."""
        return candidate == self.endpoint and budget == self.endpoint_budget
