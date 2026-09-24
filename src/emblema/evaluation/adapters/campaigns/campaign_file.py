"""The file a campaign is declared in, and the sections it is made of.

One class per section rather than one flat table, because the sections are what a comparison is
made of: who competes, at which budgets, and what a verdict will require of them. They are read
together and never apart, so they live together here.

Pydantic rather than a dataclass because a file is parsed, not constructed: every key is
checked, a key nobody reads is refused, and the sections are frozen once read.
"""

import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict

from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.statistics.comparison_rules import ComparisonRules
from emblema.evaluation.domain.statistics.holm_correction import HolmCorrection
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.domain.tuning.tuned_choice import TunedChoice
from emblema.shared.kernel.compute import ComputeTier


class _Section(BaseModel):
    """What every section of the file is: fixed once read, and closed to keys nobody reads."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class _Candidates(_Section):
    """Who competes, which of them the rest are measured against, and which carries the claim.

    Attributes:
        competing: Every candidate, in reporting order, the control among them.
        control: The one the others are measured against.
        endpoint: The one the campaign's single claim is about.
    """

    competing: tuple[str, ...]
    control: str
    endpoint: str


class _Budgets(_Section):
    """How many labels each cell learns from, at which of them the claim is made, and the repeats.

    The budgets are text, because one of them is the word for every window the tuning side
    holds rather than a count, and a file that wrote that one as a number would be writing a
    different budget on every corpus.

    Attributes:
        labels: Budgets the curve is read at, each a count or the word for all of them.
        endpoint: The budget the claim is made at; one of the above.
        seeds: Repeats of every cell; the first is the one whose fitted candidate is kept.
    """

    labels: tuple[str, ...]
    endpoint: str
    seeds: tuple[int, ...]


class _Rules(_Section):
    """What a verdict requires, as the campaign's registration states it.

    Attributes:
        minimum_relative_reduction: Least share of the control's error a candidate must remove
            for the difference to be called one.
        floor_share: Share of the control's error below which a difference is not worth
            reporting whatever the interval says.
        alpha: Family-wise error rate the Holm correction is applied at.
        secondary_family_size: How many comparisons the secondary family holds.
    """

    minimum_relative_reduction: float
    floor_share: float
    alpha: float = 0.05
    secondary_family_size: int


class _Bootstrap(_Section):
    """How the interval around each comparison is drawn.

    Attributes:
        resamples: Draws of the paired units per comparison.
        seed: Seed of the draws; repeats give the same interval.
        level: Coverage of the interval.
    """

    resamples: int = 10_000
    seed: int = 1
    level: float = 0.95


class _Selection(_Section):
    """How a campaign that chooses among variants divides the tuning side.

    Attributes:
        one_in: One tuning unit in this many is held out to score the variants on.
    """

    one_in: int


class _Tuned(_Section):
    """One variant a comparison runs at one budget, and the selection that chose it.

    Attributes:
        candidate: The candidate as the comparison names it.
        budget: The budget the variant runs at.
        variant: The variant the selection chose.
        selected_by: Identifier of the finished selection campaign.
    """

    candidate: str
    budget: str
    variant: str
    selected_by: str


class CampaignFile(_Section):
    """A campaign as it is declared before anything of it runs.

    Everything but the task: which corpus the comparison is over is an argument of the run,
    because the same design is what makes two corpora comparable, while the file states what
    must not change between them.

    Attributes:
        name: What the campaign is called, for whoever reads the file.
        purpose: What it is for; only the final one may see the task's frozen side.
        tier: Hardware class every run is measured on, declared rather than inferred from
            whichever machine picks the work up.
        candidates: Who competes.
        budgets: At how many labels, and how many times.
        rules: What a verdict requires.
        bootstrap: How each interval is drawn.
        selection: How the tuning side is divided, in a campaign that selects among variants.
        tuned: Which variant each tuned pairing runs, in a campaign that compares.
    """

    name: str
    purpose: RunPurpose = RunPurpose.TUNING
    tier: ComputeTier
    candidates: _Candidates
    budgets: _Budgets
    rules: _Rules
    bootstrap: _Bootstrap = _Bootstrap()
    selection: _Selection | None = None
    tuned: tuple[_Tuned, ...] = ()

    @classmethod
    def load(cls, path: Path) -> Self:
        """The campaign declared in the file at ``path``.

        Raises:
            ValidationError: If a section is missing, malformed, or names a key nobody reads.
        """
        with path.open("rb") as handle:
            return cls.model_validate(tomllib.load(handle))

    def competing(self) -> tuple[CandidateRef, ...]:
        """Who competes, in reporting order."""
        return tuple(CandidateRef(name) for name in self.candidates.competing)

    def control(self) -> CandidateRef:
        """The candidate the others are measured against."""
        return CandidateRef(self.candidates.control)

    def endpoint(self) -> CandidateRef:
        """The candidate the campaign's single claim is about."""
        return CandidateRef(self.candidates.endpoint)

    def label_budgets(self) -> tuple[LabelBudget, ...]:
        """The budgets the curve is read at.

        Raises:
            InvalidLabelBudgetError: If one of them is neither a count nor the word for all.
        """
        return tuple(LabelBudget.parse(text) for text in self.budgets.labels)

    def endpoint_budget(self) -> LabelBudget:
        """The budget the claim is made at.

        Raises:
            InvalidLabelBudgetError: If it is neither a count nor the word for all.
        """
        return LabelBudget.parse(self.budgets.endpoint)

    def comparison_rules(self) -> ComparisonRules:
        """What a verdict requires.

        Raises:
            InvalidComparisonRulesError: If what the file declares is not a rule that stands up.
            InvalidHolmCorrectionError: If the error rate is not one.
        """
        return ComparisonRules(
            minimum_relative_reduction=self.rules.minimum_relative_reduction,
            floor_share=self.rules.floor_share,
            holm=HolmCorrection(alpha=self.rules.alpha),
            secondary_family_size=self.rules.secondary_family_size,
        )

    def paired_bootstrap(self) -> PairedUnitBootstrap:
        """How the interval around each comparison is drawn.

        Raises:
            InvalidBootstrapIntervalError: If what the file declares is not an interval.
        """
        return PairedUnitBootstrap(
            resamples=self.bootstrap.resamples,
            seed=self.bootstrap.seed,
            level=self.bootstrap.level,
        )

    def inner_holdout(self) -> InnerHoldout | None:
        """How the tuning side is divided, if this campaign selects among variants.

        Raises:
            InvalidInnerHoldoutError: If one unit in fewer than two is to be held out.
        """
        return None if self.selection is None else InnerHoldout(one_in=self.selection.one_in)

    def tuned_choices(self) -> tuple[TunedChoice, ...]:
        """Every tuned pairing the file names.

        Raises:
            InvalidCandidateRefError: If a candidate or a variant is not a name.
            InvalidLabelBudgetError: If a budget is not one.
            ValueError: If a selection is not named by a campaign's identifier.
            InvalidTunedChoiceError: If a variant is not of the candidate it is named for.
        """
        return tuple(
            TunedChoice(
                candidate=CandidateRef(tuned.candidate),
                budget=LabelBudget.parse(tuned.budget),
                variant=CandidateRef(tuned.variant),
                selected_by=CampaignId.parse(tuned.selected_by),
            )
            for tuned in self.tuned
        )
