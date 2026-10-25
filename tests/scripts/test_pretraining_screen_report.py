"""The declared reading of a screen: the spread, the levels, the contrasts and the null pairs."""

import csv
from math import sqrt
from pathlib import Path

import pytest

from emblema.evaluation.domain.statistics.outcome_level import OutcomeLevel
from scripts.campaign_pairs_report import COMPARISON_COLUMNS, COMPARISONS
from scripts.pretraining_screen_report import (
    FLOOR,
    Pair,
    Pairs,
    parse_arguments,
    read_pairs,
    render,
    screen,
)

SCRATCH = "from_scratch@bins=6,width=64"
TUNED = "full_fine_tuning"
A1, A2, B1, B2 = "a1", "a2", "b1", "b2"


def pair(
    mean: float,
    *,
    se: float = 0.005,
    half: float = 0.01,
    budget: str = "200",
    control_campaign: str = A1,
    control: str = SCRATCH,
    candidate_campaign: str = A1,
    candidate: str = TUNED,
) -> Pair:
    return Pair(
        control_campaign=control_campaign,
        control=control,
        candidate_campaign=candidate_campaign,
        candidate=candidate,
        budget=budget,
        mean=mean,
        se=se,
        low=mean - half,
        high=mean + half,
    )


def comparisons() -> list[Pair]:
    rows = []
    for budget in ("50", "200"):
        rows += [
            pair(
                0.03, budget=budget, control_campaign=f"{A1}+{A2}", candidate_campaign=f"{A1}+{A2}"
            ),
            pair(
                0.0, budget=budget, control_campaign=f"{B1}+{B2}", candidate_campaign=f"{B1}+{B2}"
            ),
            pair(
                0.06 if budget == "200" else 0.0,
                budget=budget,
                control=TUNED,
                control_campaign=f"{B1}+{B2}",
                candidate_campaign=f"{A1}+{A2}",
            ),
            pair(0.002, budget=budget, control=TUNED, control_campaign=A1, candidate_campaign=A2),
            pair(-0.004, budget=budget, control=TUNED, control_campaign=B1, candidate_campaign=B2),
            pair(
                0.001, budget=budget, candidate=SCRATCH, control_campaign=A1, candidate_campaign=B1
            ),
        ]
    return rows


def test_the_screen_reads_every_declared_pair_and_renders_the_rule() -> None:
    result = screen(
        Pairs(comparisons()),
        scratch=SCRATCH,
        tuned=TUNED,
        gains=[("a", f"{A1}+{A2}"), ("b", f"{B1}+{B2}")],
        contrasts=[("a over b", f"{B1}+{B2}", f"{A1}+{A2}")],
        spreads=[(A1, A2), (B1, B2)],
        nulls=[(A1, B1)],
        budgets=["50", "200"],
        deciding=["50", "200"],
    )

    assert result.spread.value == FLOOR
    assert result.spread.own == 0.0
    assert [g.n for g in result.gains] == [2, 2, 2, 2]
    assert [g.level for g in result.gains] == [OutcomeLevel.LEAST_GAIN, OutcomeLevel.NOISE] * 2
    assert result.nulls_hold
    assert result.moved("a over b")
    rendered = render(result)
    assert "| 200 | a | 2 | +0.0300 ± 0.0050 |" in rendered
    assert "a over b: moves the recipe." in rendered


def test_a_null_pair_outside_its_bound_stops_the_reading() -> None:
    rows = [
        pair(0.05, budget=p.budget, candidate=SCRATCH, control_campaign=A1, candidate_campaign=B1)
        if p.candidate == SCRATCH
        else p
        for p in comparisons()
    ]

    result = screen(
        Pairs(rows),
        scratch=SCRATCH,
        tuned=TUNED,
        gains=[("a", f"{A1}+{A2}")],
        contrasts=[],
        spreads=[(A1, A2)],
        nulls=[(A1, B1)],
        budgets=["50", "200"],
        deciding=["200"],
    )

    assert not result.nulls_hold
    rendered = render(result)
    assert "OUTSIDE" in rendered
    assert "the reading stops here" in rendered
    assert "Gains" not in rendered


def test_a_declared_pair_never_compared_is_refused() -> None:
    with pytest.raises(ValueError, match="no pair of"):
        screen(
            Pairs(comparisons()),
            scratch=SCRATCH,
            tuned=TUNED,
            gains=[("a", A1)],
            contrasts=[],
            spreads=[(A1, A2)],
            nulls=[],
            budgets=["200"],
            deciding=["200"],
        )


def test_the_pairs_are_read_from_the_comparisons_the_pairs_report_wrote(tmp_path: Path) -> None:
    with (tmp_path / COMPARISONS).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS, restval="")
        writer.writeheader()
        writer.writerow(
            {
                "control_campaign": A1,
                "control": SCRATCH,
                "candidate_campaign": A1,
                "candidate": TUNED,
                "budget": "200",
                "seed_reduction_mean": "0.03",
                "seed_reduction_se": "0.005",
                "low": "0.02",
                "high": "0.04",
            }
        )

    [found] = read_pairs(tmp_path / COMPARISONS)

    assert found == Pair(
        control_campaign=A1,
        control=SCRATCH,
        candidate_campaign=A1,
        candidate=TUNED,
        budget="200",
        mean=0.03,
        se=0.005,
        low=0.02,
        high=0.04,
    )


def test_the_command_line_needs_something_to_read_and_deciding_budgets_it_reads() -> None:
    common = ["--comparisons", "d", "--scratch", SCRATCH, "--tuned", TUNED, "--spread", A1, A2]
    with pytest.raises(SystemExit):
        parse_arguments(common)
    with pytest.raises(SystemExit):
        parse_arguments([*common, "--gain", "a", A1, "--budgets", "200", "--deciding", "50"])

    arguments = parse_arguments([*common, "--gain", "a", f"{A1}+{A2}"])

    assert arguments.gain == [["a", f"{A1}+{A2}"]]
    assert arguments.deciding == ["50", "200"]


def test_the_spread_pools_the_pairs_of_every_budget_read() -> None:
    rows = [
        pair(0.0, budget="50", control=TUNED, control_campaign=A1, candidate_campaign=A2),
        pair(0.05, budget="200", control=TUNED, control_campaign=A1, candidate_campaign=A2),
        pair(0.03, budget="50", control_campaign=A1, candidate_campaign=A1),
        pair(0.03, budget="200", control_campaign=A1, candidate_campaign=A1),
    ]
    noise = 0.005**2 + (0.01 / 1.96) ** 2

    result = screen(
        Pairs(rows),
        scratch=SCRATCH,
        tuned=TUNED,
        gains=[("a", A1)],
        contrasts=[],
        spreads=[(A1, A2)],
        nulls=[],
        budgets=["50", "200"],
        deciding=["200"],
    )

    assert result.spread.own == pytest.approx(sqrt((0.05**2 - 2 * noise) / 2))
    assert {g.spread for g in result.gains} == {result.spread}
