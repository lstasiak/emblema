import tomllib
from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError

from scripts.corpus_budget_report import (
    DEFAULT_CONFIG,
    Backbone,
    Budget,
    Campaign,
    Corpus,
    CostModel,
    Estimate,
    Licence,
    Measured,
    MeasuredWindow,
    Window,
    backbone_hours,
    campaign_run_hours,
    flops_per_window,
    gpu_hours,
    parameter_count,
    pretraining_hours,
    render,
    window_stats,
    windows_in_series,
)

SAMPLES = Window(name="w50s5", unit="samples", length=50, stride=5, default=True)
HOURS = Window(name="h24s12", unit="hours", length=24, stride=12, default=True)
COST = CostModel(
    flops_per_parameter_token=6, attention_flops_coefficient=12, estimate_error_factor=3
)
LICENCE = Licence(
    terms="CC BY 4.0",
    url="https://example.org",
    checked_on=date(2026, 9, 11),
    publish_results="yes",
    sample_in_repo="yes",
    derivatives="yes",
)


def corpus(**overrides: object) -> Corpus:
    fields: dict[str, object] = {
        "name": "Test corpus",
        "role": "test",
        "regime": "regular",
        "publisher": "nobody",
        "source": "https://example.org/data.zip",
        "download_size": "1 MB",
        "licence": LICENCE,
        "epochs": 10,
        "pretraining_split": "all of it",
        "windows": (SAMPLES,),
        "estimate": Estimate(units=10, channels=21, series_length=226, basis="made up"),
        "verdict": "pretraining",
        "decision": "test",
    }
    fields.update(overrides)
    return Corpus.model_validate(fields)


def test_parameter_count_reproduces_the_reference_shape():
    assert parameter_count(256, 6) == pytest.approx(4.72e6, rel=0.01)


@pytest.mark.parametrize(("length", "expected"), [(49, 0), (50, 1), (54, 1), (55, 2), (226, 36)])
def test_windows_in_series_counts_full_windows_only(length: int, expected: int):
    assert windows_in_series(length, SAMPLES) == expected


def test_flops_per_window_adds_the_dense_and_the_quadratic_term():
    dense = 6 * 1e6 * 1000
    attention = 12 * 1000**2 * 256 * 6
    assert flops_per_window(1000, 1e6, 256, 6, COST) == pytest.approx(dense + attention)


def test_gpu_hours_divides_by_the_sustained_throughput():
    assert gpu_hours(3.6e16, 10.0) == pytest.approx(1.0)


def test_estimate_of_a_regular_corpus_multiplies_windows_by_channels():
    stats = window_stats("x", corpus(), SAMPLES)

    assert (stats.count, stats.tokens_per_window, stats.basis) == (10 * 36, 50 * 21, "estimate")


def test_estimate_of_an_irregular_corpus_scales_observations_to_the_window():
    irregular = corpus(
        regime="irregular",
        windows=(HOURS,),
        estimate=Estimate(
            units=4,
            channels=37,
            series_length=48,
            observations_per_unit=480,
            timeless_tokens_per_unit=5,
            basis="made up",
        ),
    )

    stats = window_stats("x", irregular, HOURS)

    assert (stats.count, stats.tokens_per_window) == (4 * 3, 480 * 0.5 + 5)


def test_unique_observations_ignore_window_overlap():
    assert corpus().unique_observations() == 10 * 226 * 21


def test_effective_observations_credit_repetition_up_to_the_cap():
    ten_epochs = corpus(epochs=10)
    two_epochs = corpus(epochs=2)

    assert ten_epochs.effective_observations(4) == 4 * ten_epochs.unique_observations()
    assert two_epochs.effective_observations(4) == 2 * two_epochs.unique_observations()


def test_native_observations_fall_back_to_unique_when_nothing_is_subsampled():
    plain = corpus()
    decimated = corpus(
        regime="irregular",
        windows=(HOURS,),
        estimate=Estimate(
            units=2,
            channels=17,
            series_length=48,
            observations_per_unit=100,
            native_observations=5000,
            basis="made up",
        ),
    )

    assert plain.native_observations() == plain.unique_observations()
    assert (decimated.native_observations(), decimated.unique_observations()) == (5000, 200)


def test_measured_windows_take_precedence_over_the_estimate():
    measured = corpus(
        measured=Measured(
            measured_on=date(2026, 9, 12),
            units=7,
            channels=21,
            observations=1_000,
            archive_bytes=3 * 2**20,
            timeless_tokens_per_unit=2,
            windows=(MeasuredWindow(name="w50s5", count=100, tokens=105_000),),
        )
    )

    stats = window_stats("x", measured, SAMPLES)

    assert (stats.count, stats.tokens_per_window, stats.basis) == (100, 1052, "measured")
    assert (measured.units, measured.unique_observations(), measured.archive_size()) == (
        7,
        1_000,
        "3 MiB",
    )


def test_longest_window_is_chosen_by_length_not_by_default_flag():
    long = Window(name="w100s10", unit="samples", length=100, stride=10)

    assert corpus(windows=(SAMPLES, long)).pick_window("longest") is long


def test_a_corpus_needs_exactly_one_default_window():
    with pytest.raises(ValidationError, match="exactly one window variant"):
        corpus(
            windows=(SAMPLES, Window(name="w1", unit="samples", length=1, stride=1, default=True))
        )


def test_an_irregular_estimate_needs_observations_per_unit():
    with pytest.raises(ValidationError, match="observations_per_unit"):
        corpus(regime="irregular", windows=(HOURS,))


@pytest.mark.parametrize("measured_names", [("other",), ("w50s5", "other"), ()])
def test_a_measured_corpus_measures_exactly_its_declared_variants(measured_names: tuple[str, ...]):
    with pytest.raises(ValidationError, match="must match the declared variants"):
        corpus(
            measured=Measured(
                measured_on=date(2026, 9, 12),
                units=1,
                channels=1,
                observations=1,
                windows=tuple(
                    MeasuredWindow(name=name, count=1, tokens=1) for name in measured_names
                ),
            )
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("campaigns", 0, "task_corpus"), "nope", "unknown corpus nope"),
        (("backbones", 0, "corpora"), ["cmapss", "nope"], "unknown corpora"),
        (("tiers", 1, "name"), "X", "tier M is the reference"),
    ],
)
def test_budget_rejects_dangling_references(
    path: tuple[str | int, ...], value: object, message: str
):
    with DEFAULT_CONFIG.open("rb") as source:
        raw = tomllib.load(source)
    target: Any = raw
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = value

    with pytest.raises(ValidationError, match=message):
        Budget.model_validate(raw)


def test_unknown_keys_in_the_configuration_are_rejected():
    with pytest.raises(ValidationError, match="extra"):
        corpus(colour="blue")


def test_backbone_hours_sum_named_corpora_and_price_leave_one_out_as_the_mean_fold():
    budget = Budget.load(DEFAULT_CONFIG)
    tier = budget.tier("M")
    per_corpus = {
        key: pretraining_hours(key, budget.corpora[key], tier, budget) for key in ("cmapss", "smd")
    }
    both = sum(per_corpus.values())

    named = backbone_hours(Backbone(name="two", corpora=("cmapss", "smd")), budget, tier)
    folds = backbone_hours(
        Backbone(name="loco", corpora=("cmapss", "smd"), leave_one_out=True, count=2), budget, tier
    )

    assert named == pytest.approx(both)
    assert folds == pytest.approx(2 * both / 2)


def test_a_full_budget_campaign_uses_every_window_of_the_task_corpus():
    budget = Budget.load(DEFAULT_CONFIG)
    model = budget.tier("M")
    task = budget.corpora["cmapss"]
    windows = task.estimate.units * windows_in_series(
        task.estimate.series_length, task.pick_window("default")
    )
    fixed = Campaign(name="fixed", runs=1, budget=windows, epochs=30, task_corpus="cmapss")
    everything = Campaign(name="all", runs=1, budget="all", epochs=30, task_corpus="cmapss")

    assert campaign_run_hours(everything, budget, model, 10.0) == pytest.approx(
        campaign_run_hours(fixed, budget, model, 10.0)
    )


def test_the_shipped_configuration_loads_and_renders_every_corpus():
    budget = Budget.load(DEFAULT_CONFIG)

    report = render(budget)

    assert "### Totals" in report
    assert all(corpus.name in report for corpus in budget.corpora.values())
