"""Turn the corpus facts and the cost assumptions into the pretraining budget tables.

Every number that drives the budget lives in ``corpus_budget.toml`` next to this script: measured
corpus facts, window variants, compute tiers, the cost model and the eligibility thresholds. This
script only does the arithmetic and prints markdown for the verification note, so a later
measurement corrects the file, not the code.

    uv run scripts/corpus_budget_report.py
    uv run scripts/corpus_budget_report.py --config path/to/other.toml

The formulas are deliberately coarse. FLOPs per window are the dense term (parameters × tokens)
plus the quadratic attention term; GPU-hours divide by a sustained throughput per device. The
report carries the error band the assumptions deserve rather than more digits.
"""

import argparse
import io
import math
import sys
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG = Path(__file__).resolve().with_name("corpus_budget.toml")

Answer = Literal["yes", "no", "unclear"]
Verdict = Literal["pretraining", "ingredient", "downstream-only", "candidate", "rejected"]
MIXED = frozenset({"pretraining", "ingredient"})
WindowChoice = Literal["default", "longest"]
Basis = Literal["measured", "estimate"]


class Strict(BaseModel):
    """Unknown keys are errors: a typo in the TOML fails instead of silently changing a budget."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class CostModel(Strict):
    flops_per_parameter_token: float = Field(
        gt=0, description="Dense FLOPs per parameter per token, forward and backward together"
    )
    attention_flops_coefficient: float = Field(
        gt=0, description="Coefficient of the quadratic term tokens² × width × layers"
    )
    estimate_error_factor: float = Field(
        ge=1, description="Multiplicative band every GPU-hour figure is quoted with"
    )


class Thresholds(Strict):
    min_training_units: int = Field(
        gt=0, description="Independent units a corpus must keep on the training side of the split"
    )
    unique_tokens_per_parameter: float = Field(
        gt=0,
        description=(
            "Distinct observed values on the pretraining side per model parameter; a value seen "
            "by several overlapping windows counts once"
        ),
    )
    effective_repetitions: float = Field(
        ge=1,
        description=(
            "Epochs of repetition credited as if they were fresh data when judging the parameter "
            "budget; beyond it repeated data is assumed to add little"
        ),
    )
    gpu_hours_cap: float = Field(
        gt=0, description="T4-class total above which the scale knobs are turned now, not later"
    )


class Tier(Strict):
    name: str = Field(description="S, M or L")
    device: str = Field(description="Hardware and precision the tier runs on")
    tflops: float = Field(gt=0, description="Sustained throughput assumed for the device, TFLOP/s")
    width: int = Field(gt=0, description="Model width")
    heads: int = Field(gt=0, description="Attention heads per block")
    layers: int = Field(gt=0, description="Encoder layers")
    feedforward_width: int = Field(
        gt=0, description="Hidden width of the feed-forward network in each block"
    )
    time_frequencies: int = Field(gt=0, description="Fourier frequencies of the time encoding")
    corpus_fraction: float = Field(gt=0, le=1, description="Share of each corpus seen per epoch")
    window: WindowChoice = Field(description="Window variant of each corpus the tier trains on")

    @property
    def parameters(self) -> float:
        return parameter_count(self.width, self.layers)


class Window(Strict):
    """One windowing scenario of a corpus.

    Regular corpora window by sample count, irregular ones by elapsed time, so ``unit`` says what
    ``length`` and ``stride`` are measured in.
    """

    name: str
    unit: Literal["samples", "hours"]
    length: float = Field(gt=0)
    stride: float = Field(gt=0)
    default: bool = False


class MeasuredWindow(Strict):
    name: str = Field(description="Matches a window variant of the corpus")
    count: int = Field(ge=0, description="Windows over the pretraining units")
    tokens: int = Field(
        ge=0, description="Observed tokens summed over those windows, timeless excluded"
    )


class Measured(Strict):
    """Facts counted from the raw files by ``corpus_facts.py``; pasted, never typed."""

    measured_on: date
    units: int = Field(ge=0, description="Independent units on the pretraining side")
    channels: int = Field(ge=0)
    observations: int = Field(
        ge=0, description="Observed values on the pretraining side, after any subsampling"
    )
    native_observations: int | None = Field(
        default=None, description="Observed values before subsampling; omitted when none is applied"
    )
    archive_bytes: int | None = Field(default=None, description="Downloaded archives, summed")
    timeless_tokens_per_unit: int = Field(default=0, ge=0)
    notes: str = ""
    windows: tuple[MeasuredWindow, ...]


class Estimate(Strict):
    """Facts from the literature, used until the corpus is measured."""

    units: int = Field(ge=0)
    channels: int = Field(ge=0)
    series_length: float = Field(gt=0, description="Mean unit length, in the corpus's window unit")
    observations_per_unit: float | None = Field(
        default=None,
        description="Irregular corpora only: mean observed values per unit, after any subsampling",
    )
    native_observations: float | None = Field(
        default=None, description="Observed values before subsampling; omitted when none is applied"
    )
    timeless_tokens_per_unit: int = Field(default=0, ge=0)
    basis: str = Field(description="Where the numbers come from")


class Licence(Strict):
    terms: str = Field(description="What the source of record says, in one or two sentences")
    url: str
    checked_on: date
    publish_results: Answer
    sample_in_repo: Answer = Field(description="A miniature sample may sit in a public repository")
    derivatives: Answer = Field(description="A tokenised corpus may be redistributed publicly")


class Subsampling(Strict):
    """How a corpus too large to tokenise whole is cut down before counting."""

    channel_set: Literal["lightweight", "target", "all"]
    min_spacing_seconds: float = Field(
        gt=0, description="At most one observation per channel per bin of this width"
    )
    lightweight: dict[str, tuple[int, int]] = Field(
        default_factory=dict, description="Per mission: inclusive channel number range"
    )
    portion: Literal["first-half", "all"] = "first-half"
    missions: tuple[str, ...] | None = Field(
        default=None,
        description="Missions that count as units; others are measured and reported in notes only",
    )


class Corpus(Strict):
    name: str
    role: str = Field(description="Role in the study, in the words of the corpus table")
    regime: Literal["regular", "irregular"]
    publisher: str
    source: str = Field(description="Download URL at the source of record")
    download_size: str = Field(description="Size at the source of record, as published")
    licence: Licence
    epochs: int = Field(gt=0, description="Pretraining epochs assumed for the budget")
    pretraining_split: str = Field(description="Which part of the corpus pretraining may see")
    windows: tuple[Window, ...] = Field(min_length=1)
    estimate: Estimate
    measured: Measured | None = None
    subsampling: Subsampling | None = None
    verdict: Verdict
    decision: str = Field(description="One sentence")

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if sum(window.default for window in self.windows) != 1:
            raise ValueError(f"{self.name}: exactly one window variant must be the default")
        names = {window.name for window in self.windows}
        if self.measured is not None:
            measured = {window.name for window in self.measured.windows}
            if measured != names:
                # A measured corpus is measured whole: otherwise one row would mix counted and
                # estimated numbers without saying so.
                raise ValueError(
                    f"{self.name}: measured windows {sorted(measured)} must match the declared "
                    f"variants {sorted(names)}"
                )
        if self.regime == "irregular" and self.estimate.observations_per_unit is None:
            raise ValueError(f"{self.name}: an irregular estimate needs observations_per_unit")
        return self

    @property
    def basis(self) -> Basis:
        return "measured" if self.measured is not None else "estimate"

    @property
    def units(self) -> int:
        return self.measured.units if self.measured is not None else self.estimate.units

    def unique_observations(self) -> float:
        """Distinct observed values pretraining can see, whatever the windowing repeats."""
        if self.measured is not None:
            return float(self.measured.observations)
        estimate = self.estimate
        if self.regime == "regular":
            return estimate.units * estimate.series_length * estimate.channels
        return estimate.units * (estimate.observations_per_unit or 0.0)

    def native_observations(self) -> float:
        if self.measured is not None:
            return float(self.measured.native_observations or self.measured.observations)
        return self.estimate.native_observations or self.unique_observations()

    def effective_observations(self, repetitions: float) -> float:
        """Unique observations with the first few epochs of repetition credited as fresh data."""
        return self.unique_observations() * min(self.epochs, repetitions)

    def archive_size(self) -> str:
        if self.measured is None or self.measured.archive_bytes is None:
            return self.download_size
        size = self.measured.archive_bytes
        return f"{size / 2**30:,.1f} GiB" if size >= 2**30 else f"{size / 2**20:,.0f} MiB"

    def pick_window(self, choice: WindowChoice) -> Window:
        if choice == "longest":
            return max(self.windows, key=lambda window: window.length)
        return next(window for window in self.windows if window.default)


class Backbone(Strict):
    name: str
    corpora: tuple[str, ...] | Literal["eligible", "each-eligible"] = Field(
        description=(
            "Named corpora; 'eligible' = every corpus in the mix at once; 'each-eligible' = one "
            "backbone per corpus that can stand alone"
        )
    )
    leave_one_out: bool = Field(
        default=False,
        description=(
            "Each run drops one corpus; priced as the mean over the k possible folds, "
            "(k-1)/k of the mix, until the folds are chosen"
        ),
    )
    count: int = Field(default=1, gt=0, description="How many such backbones")


class Campaign(Strict):
    name: str
    runs: int = Field(gt=0)
    budget: int | Literal["all"] = Field(description="Labelled windows per run")
    epochs: int = Field(gt=0)
    task_corpus: str


class FixedCost(Strict):
    name: str
    gpu_hours: float = Field(ge=0, description="T4-class hours, scaled to the other devices")
    basis: str


class Sweep(Strict):
    shapes: tuple[tuple[int, int], ...] = Field(description="(width, layers) pairs to price")


class Budget(Strict):
    cost_model: CostModel
    thresholds: Thresholds
    tiers: tuple[Tier, ...] = Field(min_length=1)
    corpora: dict[str, Corpus]
    backbones: tuple[Backbone, ...]
    campaigns: tuple[Campaign, ...]
    fixed_costs: tuple[FixedCost, ...] = ()
    sweep: Sweep

    @classmethod
    def load(cls, path: Path) -> Self:
        with path.open("rb") as source:
            return cls.model_validate(tomllib.load(source))

    @model_validator(mode="after")
    def check_references(self) -> Self:
        for campaign in self.campaigns:
            if campaign.task_corpus not in self.corpora:
                raise ValueError(
                    f"campaign {campaign.name!r}: unknown corpus {campaign.task_corpus}"
                )
        for backbone in self.backbones:
            if isinstance(backbone.corpora, tuple):
                unknown = set(backbone.corpora) - self.corpora.keys()
                if unknown:
                    raise ValueError(
                        f"backbone {backbone.name!r}: unknown corpora {sorted(unknown)}"
                    )
        if not any(tier.name == "M" for tier in self.tiers):
            raise ValueError("tier M is the reference for thresholds and campaigns")
        return self

    def tier(self, name: str) -> Tier:
        return next(tier for tier in self.tiers if tier.name == name)

    def eligible(self) -> list[str]:
        """Corpora that enter the pretraining mix."""
        return [key for key, corpus in self.corpora.items() if corpus.verdict in MIXED]

    def standalone(self) -> list[str]:
        """Corpora with enough independent units to carry a backbone of their own."""
        return [key for key, corpus in self.corpora.items() if corpus.verdict == "pretraining"]


# --- arithmetic -------------------------------------------------------------------------------


def parameter_count(width: int, layers: int) -> float:
    """Dense parameters of a transformer encoder: 4·d² attention + 8·d² MLP per layer."""
    return 12.0 * width**2 * layers


def windows_in_series(length: float, window: Window) -> int:
    """Full windows that fit a unit of ``length`` at the window's stride; 0 when none fits."""
    if length < window.length:
        return 0
    return math.floor((length - window.length) / window.stride) + 1


def flops_per_window(
    tokens: float, parameters: float, width: int, layers: int, cost: CostModel
) -> float:
    dense = cost.flops_per_parameter_token * parameters * tokens
    attention = cost.attention_flops_coefficient * tokens**2 * width * layers
    return dense + attention


def gpu_hours(flops: float, tflops: float) -> float:
    return flops / (tflops * 1e12 * 3600)


@dataclass(frozen=True)
class WindowStats:
    """What one window variant of a corpus yields per epoch."""

    corpus: str
    window: Window
    count: int
    tokens_per_window: float
    basis: Basis

    @property
    def tokens_per_epoch(self) -> float:
        return self.count * self.tokens_per_window


def window_stats(key: str, corpus: Corpus, window: Window) -> WindowStats:
    measured = corpus.measured
    if measured is not None:
        found = next(item for item in measured.windows if item.name == window.name)
        observed = found.tokens / found.count if found.count else 0.0
        per_window = observed + measured.timeless_tokens_per_unit
        return WindowStats(key, window, found.count, per_window, "measured")
    estimate = corpus.estimate
    count = estimate.units * windows_in_series(estimate.series_length, window)
    if corpus.regime == "regular":
        observed = window.length * estimate.channels
    else:
        per_unit = estimate.observations_per_unit or 0.0
        observed = per_unit * min(1.0, window.length / estimate.series_length)
    return WindowStats(key, window, count, observed + estimate.timeless_tokens_per_unit, "estimate")


def pretraining_hours(key: str, corpus: Corpus, tier: Tier, budget: Budget) -> float:
    stats = window_stats(key, corpus, corpus.pick_window(tier.window))
    flops = flops_per_window(
        stats.tokens_per_window, tier.parameters, tier.width, tier.layers, budget.cost_model
    )
    windows_seen = stats.count * tier.corpus_fraction * corpus.epochs
    return gpu_hours(windows_seen * flops, tier.tflops)


def backbone_corpora(backbone: Backbone, budget: Budget) -> list[list[str]]:
    """The corpus sets this backbone entry expands to, one list per backbone."""
    if backbone.corpora == "each-eligible":
        return [[key] for key in budget.standalone()]
    keys = budget.eligible() if backbone.corpora == "eligible" else list(backbone.corpora)
    return [keys for _ in range(backbone.count)]


def backbone_hours(backbone: Backbone, budget: Budget, tier: Tier) -> float:
    total = 0.0
    for keys in backbone_corpora(backbone, budget):
        hours = sum(pretraining_hours(key, budget.corpora[key], tier, budget) for key in keys)
        if backbone.leave_one_out and len(keys) > 1:
            hours *= (len(keys) - 1) / len(keys)
        total += hours
    return total


def campaign_run_hours(campaign: Campaign, budget: Budget, model: Tier, tflops: float) -> float:
    """One fine-tuning run of the ``model`` tier's backbone on a device of ``tflops``."""
    corpus = budget.corpora[campaign.task_corpus]
    stats = window_stats(campaign.task_corpus, corpus, corpus.pick_window("default"))
    windows = stats.count if campaign.budget == "all" else campaign.budget
    flops = flops_per_window(
        stats.tokens_per_window, model.parameters, model.width, model.layers, budget.cost_model
    )
    return gpu_hours(windows * campaign.epochs * flops, tflops)


def tokens_per_epoch(budget: Budget, key: str, choice: WindowChoice) -> float:
    corpus = budget.corpora[key]
    return window_stats(key, corpus, corpus.pick_window(choice)).tokens_per_epoch


def mixed_tokens_per_epoch(budget: Budget, choice: WindowChoice = "default") -> float:
    return sum(tokens_per_epoch(budget, key, choice) for key in budget.eligible())


def mixed_tokens_seen(budget: Budget, choice: WindowChoice = "default") -> float:
    return sum(
        tokens_per_epoch(budget, key, choice) * budget.corpora[key].epochs
        for key in budget.eligible()
    )


def mixed_unique_observations(budget: Budget) -> float:
    return sum(budget.corpora[key].unique_observations() for key in budget.eligible())


def mixed_effective_observations(budget: Budget) -> float:
    credit = budget.thresholds.effective_repetitions
    return sum(budget.corpora[key].effective_observations(credit) for key in budget.eligible())


# --- rendering --------------------------------------------------------------------------------


def si(value: float) -> str:
    """Compact magnitude: 1.2k, 34M, 1.6G, 2.5T."""
    for threshold, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= threshold:
            return f"{value / threshold:.3g}{suffix}"
    return f"{value:.3g}"


def hours(value: float) -> str:
    if value == 0:
        return "0"
    if value < 1 / 60:
        return f"{value * 3600:.0f} s"
    if value < 1:
        return f"{value * 60:.0f} min"
    return f"{value:.1f} h"


def table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_assumptions(budget: Budget) -> str:
    cost = budget.cost_model
    rows = [
        (
            tier.name,
            tier.device,
            f"{tier.tflops:g}",
            f"{tier.width} × {tier.layers}",
            si(tier.parameters),
            f"{tier.corpus_fraction:g}",
            tier.window,
        )
        for tier in budget.tiers
    ]
    return "\n".join(
        [
            f"FLOPs per window = {cost.flops_per_parameter_token:g} · P · n + "
            f"{cost.attention_flops_coefficient:g} · n² · d · L; P = 12 · d² · L. "
            f"Every GPU-hour below is ±{cost.estimate_error_factor:g}×.",
            "",
            table(("Tier", "Device", "TFLOP/s", "d × L", "P", "Corpus fraction", "Window"), rows),
        ]
    )


def render_summary(budget: Budget) -> str:
    rows = []
    for corpus in budget.corpora.values():
        native = corpus.native_observations()
        unique = corpus.unique_observations()
        rows.append(
            (
                corpus.name,
                corpus.role,
                corpus.regime,
                corpus.archive_size(),
                f"{corpus.units:,}",
                si(native),
                si(unique) if unique != native else "same",
                corpus.pretraining_split,
                corpus.basis,
            )
        )
    return table(
        (
            "Corpus",
            "Role",
            "Regime",
            "Download",
            "Units",
            "Observed values",
            "After subsampling",
            "Pretraining side",
            "Basis",
        ),
        rows,
    )


def render_corpora(budget: Budget) -> str:
    reference = budget.tier("M")
    rows = []
    for key, corpus in budget.corpora.items():
        for window in corpus.windows:
            stats = window_stats(key, corpus, window)
            flops = flops_per_window(
                stats.tokens_per_window,
                reference.parameters,
                reference.width,
                reference.layers,
                budget.cost_model,
            )
            per_epoch = gpu_hours(stats.count * flops, reference.tflops)
            rows.append(
                (
                    corpus.name,
                    f"{window.name}{' (default)' if window.default else ''}",
                    f"{corpus.units:,}",
                    f"{stats.count:,}",
                    f"{stats.tokens_per_window:,.0f}",
                    si(stats.tokens_per_epoch),
                    f"{corpus.epochs}",
                    hours(per_epoch * corpus.epochs),
                    stats.basis,
                )
            )
    return table(
        (
            "Corpus",
            "Variant",
            "Units",
            "Windows",
            "Tokens/window",
            "Tokens/epoch",
            "Epochs",
            f"GPU-h at {reference.name}",
            "Basis",
        ),
        rows,
    )


def render_thresholds(budget: Budget) -> str:
    reference = budget.tier("M")
    limits = budget.thresholds
    needed = limits.unique_tokens_per_parameter * reference.parameters
    rows = []
    for corpus in budget.corpora.values():
        unique = corpus.unique_observations()
        effective = corpus.effective_observations(limits.effective_repetitions)
        rows.append(
            (
                corpus.name,
                f"{corpus.units:,}",
                "yes" if corpus.units >= limits.min_training_units else "**no**",
                si(unique),
                "yes" if unique >= needed else "**no**",
                f"{effective / reference.parameters:.0f}",
                corpus.verdict,
                corpus.decision,
            )
        )
    return "\n".join(
        [
            f"Thresholds: ≥ {limits.min_training_units} training units; unique observed values ≥ "
            f"{limits.unique_tokens_per_parameter:g} × P at tier {reference.name} "
            f"(= {si(needed)}). Overlapping windows do not multiply the count. The last column "
            f"credits up to {limits.effective_repetitions:g} epochs of repetition as fresh data.",
            "",
            table(
                (
                    "Corpus",
                    "Units",
                    "Units ok",
                    "Unique values",
                    "Unique ok",
                    "Effective per P",
                    "Verdict",
                    "Decision",
                ),
                rows,
            ),
        ]
    )


def render_licences(budget: Budget) -> str:
    rows = [
        (
            corpus.name,
            corpus.publisher,
            corpus.licence.terms,
            corpus.licence.publish_results,
            corpus.licence.sample_in_repo,
            corpus.licence.derivatives,
            corpus.licence.checked_on.isoformat(),
        )
        for corpus in budget.corpora.values()
    ]
    return table(
        ("Corpus", "Publisher", "Terms", "Results", "Sample in repo", "Derivatives", "Checked"),
        rows,
    )


def render_programme(budget: Budget) -> str:
    rows = []
    totals = dict.fromkeys(budget.tiers, 0.0)
    for backbone in budget.backbones:
        sets = backbone_corpora(backbone, budget)
        label = ", ".join(sorted({key for keys in sets for key in keys})) or "—"
        per_tier = {tier: backbone_hours(backbone, budget, tier) for tier in budget.tiers}
        for tier, value in per_tier.items():
            totals[tier] += value
        rows.append(
            (
                backbone.name,
                f"{len(sets)}",
                label,
                *(hours(per_tier[tier]) for tier in budget.tiers),
            )
        )
    rows.append(("**Total**", "", "", *(f"**{hours(totals[tier])}**" for tier in budget.tiers)))
    header = (
        "Backbone",
        "Runs",
        "Corpora",
        *(f"{tier.name} ({tier.device})" for tier in budget.tiers),
    )
    return "\n".join(
        [
            f"In the mix: {', '.join(budget.eligible()) or 'none'}; with a backbone of their "
            f"own: {', '.join(budget.standalone()) or 'none'}.",
            "",
            table(header, rows),
        ]
    )


def render_campaigns(budget: Budget) -> str:
    model = budget.tier("M")
    rows = []
    totals = dict.fromkeys(budget.tiers, 0.0)
    for campaign in budget.campaigns:
        per_tier = {
            tier: campaign_run_hours(campaign, budget, model, tier.tflops) * campaign.runs
            for tier in budget.tiers
        }
        for tier, value in per_tier.items():
            totals[tier] += value
        run_minutes = campaign_run_hours(campaign, budget, model, model.tflops) * 60
        rows.append(
            (
                campaign.name,
                f"{campaign.runs}",
                str(campaign.budget),
                f"{campaign.epochs}",
                f"{run_minutes:.1f} min",
                *(hours(per_tier[tier]) for tier in budget.tiers),
            )
        )
    rows.append(
        ("**Total**", "", "", "", "", *(f"**{hours(totals[tier])}**" for tier in budget.tiers))
    )
    header = (
        "Campaign",
        "Runs",
        "Budget",
        "Epochs",
        f"Per run on {model.device}",
        *(f"Total on {tier.device}" for tier in budget.tiers),
    )
    return "\n".join(
        [
            f"Fine-tuning always uses the tier-{model.name} backbone ({si(model.parameters)} "
            "parameters); the columns differ only by device.",
            "",
            table(header, rows),
        ]
    )


def render_totals(budget: Budget) -> str:
    reference = budget.tier("M")
    band = budget.cost_model.estimate_error_factor
    rows = []
    for tier in budget.tiers:
        pretraining = sum(backbone_hours(backbone, budget, tier) for backbone in budget.backbones)
        campaigns = sum(
            campaign_run_hours(campaign, budget, reference, tier.tflops) * campaign.runs
            for campaign in budget.campaigns
        )
        fixed = sum(cost.gpu_hours for cost in budget.fixed_costs) * reference.tflops / tier.tflops
        total = pretraining + campaigns + fixed
        rows.append(
            (
                tier.name,
                hours(pretraining),
                hours(campaigns),
                hours(fixed),
                f"**{hours(total)}**",
                f"{hours(total / band)} to {hours(total * band)}",
            )
        )
    fixed_rows = [(cost.name, hours(cost.gpu_hours), cost.basis) for cost in budget.fixed_costs]
    parts = [
        f"Cap: {budget.thresholds.gpu_hours_cap:g} h of {reference.device} for the whole project.",
        "",
        table(("Tier", "Pretraining", "Campaigns", "Fixed", "Total", f"Band ±{band:g}×"), rows),
    ]
    if fixed_rows:
        parts += [
            "",
            "Fixed allowances (T4-class):",
            "",
            table(("Item", "GPU-h", "Basis"), fixed_rows),
        ]
    return "\n".join(parts)


def render_sweep(budget: Budget) -> str:
    reference = budget.tier("M")
    limits = budget.thresholds
    unique = mixed_unique_observations(budget)
    effective = mixed_effective_observations(budget)
    processed = mixed_tokens_per_epoch(budget)
    seen = mixed_tokens_seen(budget)
    rows = []
    for width, layers in budget.sweep.shapes:
        parameters = parameter_count(width, layers)
        shape = reference.model_copy(update={"width": width, "layers": layers})
        mixed = sum(
            pretraining_hours(key, budget.corpora[key], shape, budget) for key in budget.eligible()
        )
        rows.append(
            (
                f"{width} × {layers}",
                si(parameters),
                f"{unique / parameters:.0f}",
                "yes" if unique >= limits.unique_tokens_per_parameter * parameters else "**no**",
                f"{effective / parameters:.0f}",
                "yes" if effective >= limits.unique_tokens_per_parameter * parameters else "**no**",
                f"{seen / parameters:.0f}",
                hours(mixed),
            )
        )
    return "\n".join(
        [
            f"Mixed corpus (every eligible corpus, default windows): {si(unique)} unique observed "
            f"values; {si(effective)} effective with up to {limits.effective_repetitions:g} "
            f"repetitions credited; {si(processed)} tokens processed per epoch and {si(seen)} "
            "over the configured epochs (compute, not information).",
            "",
            table(
                (
                    "d × L",
                    "P",
                    "Unique per P",
                    f"≥ {limits.unique_tokens_per_parameter:g}",
                    "Effective per P",
                    f"≥ {limits.unique_tokens_per_parameter:g}",
                    "Seen per P",
                    f"Mixed pretraining on {reference.device}",
                ),
                rows,
            ),
        ]
    )


def render(budget: Budget) -> str:
    sections = (
        ("Assumptions", render_assumptions(budget)),
        ("Corpora", render_summary(budget)),
        ("Window variants", render_corpora(budget)),
        ("Eligibility", render_thresholds(budget)),
        ("Licences", render_licences(budget)),
        ("Pretraining programme", render_programme(budget)),
        ("Fine-tuning campaigns", render_campaigns(budget)),
        ("Totals", render_totals(budget)),
        ("Parameter budget", render_sweep(budget)),
    )
    return "\n\n".join(f"### {title}\n\n{body}" for title, body in sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if isinstance(sys.stdout, io.TextIOWrapper):
        # The tables use ×, ≥ and ²; a Windows console defaults to a code page that lacks them, and
        # the note this output is pasted into is LF-only.
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    print(render(Budget.load(args.config)), end="")


if __name__ == "__main__":
    main()
