"""The budget read off a pretraining run's log: what it parses, the rule, and what it writes."""

from pathlib import Path

import pytest

from scripts.pretraining_budget_report import (
    KEPT_LAG,
    CostRow,
    budget_of,
    costs_of,
    mean_costs,
    parse_arguments,
    read_costs,
    render,
    stops_learning,
    write_costs,
)

PREFIX = (
    "2026-10-24 23:25:51,554 INFO emblema.pretraining.adapters.training.torch_training_runtime: "
)
# A line as the runtime writes it, two corpora with the cells' terms (the pilot over both).
PILOT = (
    PREFIX + "epoch 2 of 2 done in 40 s, 83 s into this session: training 0.64203 (reconstruction "
    "0.18851, presence 0.44475, count 0.00877); physionet2012 validation 0.21130, 1.003 of the "
    "trivial predictor's (reconstruction 0.21130 of 0.21065, presence 0.54335 of 0.60492, count "
    "0.02948 of 0.04529); physionet2019 validation 0.18532, 0.957 of the trivial predictor's "
    "(reconstruction 0.18532 of 0.19364, presence 0.43086 of 0.60409, count 0.00617 of 0.01593)"
)


def line(epoch: int, cap: int, costs: dict[str, float]) -> str:
    """An epoch's line over one corpus whose trivial predictor costs 1 for every term."""
    terms = ", ".join(f"{name} {cost:.5f} of 1.00000" for name, cost in costs.items())
    return (
        f"{PREFIX}epoch {epoch} of {cap} done in 10 s, 10 s into this session: training 0.5; "
        f"stays validation {costs['reconstruction']:.5f}, {costs['reconstruction']:.3f} of the "
        f"trivial predictor's ({terms})"
    )


def run(values: list[float], presence: list[float]) -> list[str]:
    return [
        line(epoch, len(values), {"reconstruction": v, "presence": p})
        for epoch, (v, p) in enumerate(zip(values, presence, strict=True), start=1)
    ]


def test_a_line_of_the_runtime_gives_every_term_of_every_corpus() -> None:
    rows = costs_of(["something else", PILOT])

    assert len(rows) == 6
    assert {row.corpus for row in rows} == {"physionet2012", "physionet2019"}
    presence = mean_costs(rows)["presence"][2]
    assert presence == pytest.approx((0.54335 / 0.60492 + 0.43086 / 0.60409) / 2)
    assert {row.epoch for row in rows} == {2}


def test_an_epoch_reported_again_after_a_resumption_keeps_its_last_report() -> None:
    first = line(1, 3, {"reconstruction": 0.9})
    again = line(1, 3, {"reconstruction": 0.8})

    [row] = costs_of([first, again])

    assert row.loss == pytest.approx(0.8)


def test_a_corpus_without_its_terms_is_refused_rather_than_read() -> None:
    bare = (
        f"{PREFIX}epoch 1 of 2 done in 1 s, 1 s into this session: training 0.5; "
        "stays validation 0.5, 0.900 of the trivial predictor's"
    )

    with pytest.raises(ValueError, match="no terms"):
        costs_of([bare])


def test_a_term_stops_learning_where_the_next_epochs_leave_it_standing() -> None:
    falling_then_flat = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.698, 5: 0.697, 6: 0.699, 7: 0.695}

    assert stops_learning(falling_then_flat) == 3


def test_a_term_still_falling_at_the_cap_has_not_stopped() -> None:
    assert stops_learning({e: 1.0 - 0.05 * e for e in range(1, 10)}) is None
    # Flat, but too few epochs after it to say so.
    assert stops_learning({1: 1.0, 2: 0.5, 3: 0.5, 4: 0.5}) is None


def test_a_fall_of_exactly_the_tolerance_does_not_count_as_learning() -> None:
    # Halves are exact in binary, so the boundary is the rule's and not the rounding's.
    assert stops_learning({1: 1.0, 2: 0.5, 3: 0.75, 4: 0.5}, tolerance=0.5) == 1
    assert stops_learning({1: 1.0, 2: 0.25, 3: 0.75, 4: 0.5, 5: 0.5}, tolerance=0.5) == 2


def test_the_budget_is_the_latest_stop_and_the_kept_epoch_follows_the_values() -> None:
    values = [1.0, 0.9, 0.88, 0.879, 0.878, 0.879, 0.88, 0.881, 0.882]
    presence = [1.0, 0.8, 0.7, 0.6, 0.55, 0.53, 0.528, 0.527, 0.526]

    budget = budget_of(costs_of(run(values, presence)))

    assert budget.stops == {"presence": 6, "reconstruction": 3}
    assert budget.epochs == 6
    assert budget.kept == 5
    assert budget.lag == 1
    assert not budget.keeps_too_early


def test_a_kept_epoch_long_before_the_presence_stops_is_named() -> None:
    values = [1.0, 0.9, 0.9, 0.9, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96]
    presence = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.449, 0.448, 0.447, 0.447]

    budget = budget_of(costs_of(run(values, presence)))

    assert budget.kept == 2
    assert budget.stops["presence"] == 7
    assert budget.lag == 5 > KEPT_LAG
    assert budget.keeps_too_early
    assert "would leave the presence unlearnt" in render(costs_of(run(values, presence)))


def test_a_term_that_learns_to_the_cap_leaves_no_budget() -> None:
    values = [1.0, 0.9, 0.89, 0.889, 0.888, 0.888, 0.888]
    presence = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

    budget = budget_of(costs_of(run(values, presence)))

    assert budget.epochs is None
    assert budget.stops["presence"] is None
    assert budget.lag is None
    assert "too short" in render(costs_of(run(values, presence)))


def test_rows_without_the_values_are_refused() -> None:
    with pytest.raises(ValueError, match="reconstruction"):
        budget_of([CostRow(1, "stays", "presence", 0.5, 1.0)])


def test_the_costs_round_trip_and_render_the_budget(tmp_path: Path) -> None:
    rows = costs_of(run([1.0, 0.9, 0.9, 0.9, 0.9], [1.0, 0.8, 0.8, 0.8, 0.8]))
    write_costs(tmp_path / "costs.csv", rows)

    assert read_costs(tmp_path / "costs.csv") == rows
    rendered = render(rows)
    assert "budget: 2 epochs" in rendered
    assert "| 1 | 1.0000 | 1.0000 |" in rendered


def test_reading_needs_a_log_and_rendering_a_directory() -> None:
    with pytest.raises(SystemExit):
        parse_arguments([])
    assert parse_arguments(["--log", "run.log"]).log == Path("run.log")
    assert parse_arguments(["--report-only", "d"]).report_only == Path("d")
