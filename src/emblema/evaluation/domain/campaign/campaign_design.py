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
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from emblema.evaluation.domain.tuning.tuned_choice import TunedChoice


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

    A candidate may run as a tuned variant at some budgets: the pairing is still the
    candidate's, and the curve still reads it under its own name, but the cell runs the variant
    a selection chose for that budget.

    Invariants: at least two candidates, none named twice; the control and the endpoint are
    among them and are not the same; at least one budget and one seed, none repeated; the
    endpoint's budget is one of the budgets; every candidate that shares the compute budget
    declares the same one; the secondary comparisons the grid holds fit the registered family;
    every tuned pairing is of a candidate and a budget of the grid, none twice, and its variant
    is described among the variants, of the same kind and compute budget as its base.

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
        inner_holdout: How the tuning side is divided, for a campaign that selects among
            variants; ``None`` for one that compares.
        tuned: Which variant a candidate runs at a budget, for each pairing a selection chose.
        variants: Every variant ``tuned`` names, as it was described when the design was
            written, so a cell is checked against the variant it runs rather than its base.
    """

    candidates: tuple[CampaignCandidate, ...]
    control: CandidateRef
    endpoint: CandidateRef
    budgets: tuple[LabelBudget, ...]
    endpoint_budget: LabelBudget
    seeds: tuple[int, ...]
    rules: ComparisonRules
    bootstrap: PairedUnitBootstrap
    inner_holdout: InnerHoldout | None = None
    tuned: tuple[TunedChoice, ...] = ()
    variants: tuple[CampaignCandidate, ...] = ()

    def __post_init__(self) -> None:
        self._check_candidates()
        self._check_axes()
        self._check_family()
        self._check_tuned()

    def _check_tuned(self) -> None:
        pairings = [(choice.candidate, choice.budget) for choice in self.tuned]
        if len(set(pairings)) != len(pairings):
            raise InvalidCampaignDesignError("a pairing is tuned twice")
        described = {variant.ref: variant for variant in self.variants}
        if len(described) != len(self.variants):
            raise InvalidCampaignDesignError("a variant is described twice")
        for choice in self.tuned:
            if choice.budget not in self.budgets:
                raise InvalidCampaignDesignError(
                    f"{choice.candidate} is tuned at {choice.budget}, a budget the grid lacks"
                )
            base = self.get_candidate(choice.candidate)
            variant = described.get(choice.variant)
            if variant is None:
                raise InvalidCampaignDesignError(f"{choice.variant} is tuned but not described")
            if (variant.kind, variant.budget) != (base.kind, base.budget):
                raise InvalidCampaignDesignError(
                    f"{choice.variant} is not of the kind and compute budget of {base.ref}"
                )
        if {variant.ref for variant in self.variants} != {c.variant for c in self.tuned}:
            raise InvalidCampaignDesignError("a variant is described that no pairing runs")

    def declared_for(self, cell: CampaignCell) -> CampaignCandidate:
        """The candidate a cell runs: the variant its pairing was tuned to, or the base.

        Raises:
            UnknownCandidateError: If the design names no such candidate.
        """
        base = self.get_candidate(cell.candidate)
        for choice in self.tuned:
            if choice.candidate == cell.candidate and choice.budget == cell.budget:
                return next(v for v in self.variants if v.ref == choice.variant)
        return base

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
        A selection keeps nothing: what it produces is a choice, and the comparison that runs
        the chosen variant keeps its own.
        """
        if self.inner_holdout is not None:
            return False
        return cell.budget == self.endpoint_budget and cell.seed == self.reference_seed

    def is_endpoint(self, candidate: CandidateRef, budget: LabelBudget) -> bool:
        """Whether this pairing is the single claim the campaign was designed to test."""
        return candidate == self.endpoint and budget == self.endpoint_budget
