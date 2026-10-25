"""The rate an arm keeps: the rate in force unless another's gain clears one paired error."""

import csv
import random
from pathlib import Path
from statistics import stdev

import pytest

from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.statistics.paired_unit_rankings import PairedUnitRankings
from scripts.campaign_pairs_report import Answers, PredictionRow, Side
from scripts.rate_choice_report import (
    CHOICES,
    Arm,
    arms_of,
    choose,
    parse_arguments,
    rate_of,
    render,
    write_choices,
)

FIRST = "596849cd-fea8-4520-800f-be0f0e117ee0"
SECOND = "0af04f55-5c2e-4aa9-8d60-c8ca6e52b088"
IN_FORCE = "full_fine_tuning"
SLOW = "full_fine_tuning@learning_rate=0.0001"
FAST = "full_fine_tuning@learning_rate=0.003"
BUDGETS = ("20", "200")
SEEDS = (1, 2)
BOOTSTRAP = PairedUnitBootstrap(resamples=200, seed=1)


def answers_of(
    campaign: str, candidate: str, budget: str, seed: int, *, signal: float, units: int = 140
) -> list[PredictionRow]:
    """One repeat: every seventh stay dies, and the answer leans towards it by ``signal``."""
    draws = random.Random(f"{campaign}/{candidate}/{budget}/{seed}")
    return [
        PredictionRow(
            campaign=campaign,
            candidate=candidate,
            budget=budget,
            seed=seed,
            unit=f"set-a/{index:05d}",
            position=0,
            ends_at=48.0,
            target=1.0 if index % 7 == 0 else 0.0,
            predicted=draws.gauss(signal if index % 7 == 0 else 0.0, 1.0),
        )
        for index in range(units)
    ]


def grid(signals: dict[str, float], campaigns: tuple[str, ...] = (FIRST,)) -> Answers:
    return Answers(
        row
        for campaign in campaigns
        for candidate, signal in signals.items()
        for budget in BUDGETS
        for seed in SEEDS
        for row in answers_of(campaign, candidate, budget, seed, signal=signal)
    )


def arm(*others: str, campaigns: tuple[str, ...] = (FIRST,)) -> Arm:
    return Arm(label="fine-tuning", campaigns=campaigns, in_force=IN_FORCE, others=others)


def chosen(answers: Answers, the_arm: Arm) -> str:
    choice = choose(answers, the_arm, rate_in_force=0.001, budgets=BUDGETS, bootstrap=BOOTSTRAP)
    return choice.chosen.candidate


def test_a_candidate_states_its_rate_or_runs_at_the_rate_in_force() -> None:
    assert rate_of(SLOW, 0.001) == 0.0001
    assert rate_of("from_scratch@bins=6,learning_rate=0.003,width=64", 0.001) == 0.003
    assert rate_of(IN_FORCE, 0.001) == 0.001
    assert rate_of("from_scratch@bins=6,width=64", 0.001) == 0.001
    with pytest.raises(ValueError, match="states its rate"):
        rate_of("full_fine_tuning@learning_rate=fast", 0.001)


def test_a_rate_that_clearly_does_better_is_chosen_and_named_at_the_edge() -> None:
    answers = grid({IN_FORCE: 0.5, SLOW: 0.5, FAST: 2.5})

    choice = choose(
        answers, arm(SLOW, FAST), rate_in_force=0.001, budgets=BUDGETS, bootstrap=BOOTSTRAP
    )

    fast = choice.chosen
    assert fast.candidate == FAST
    assert fast.repeats == len(BUDGETS) * len(SEEDS)
    assert fast.gain == pytest.approx(fast.area - choice.in_force.area)
    assert fast.gain > fast.standard_error > 0.0
    assert choice.at_edge
    assert "chosen rate: 0.003 (full_fine_tuning@learning_rate=0.003) — at an edge" in render(
        [choice]
    )


def test_the_rate_in_force_stays_when_no_other_clears_its_error() -> None:
    answers = grid({IN_FORCE: 1.0, SLOW: 1.0, FAST: 1.0})

    choice = choose(
        answers, arm(SLOW, FAST), rate_in_force=0.001, budgets=BUDGETS, bootstrap=BOOTSTRAP
    )

    assert choice.chosen is choice.in_force
    assert not choice.at_edge
    assert "chosen rate: 0.001 (the rate in force)." in render([choice])


def test_of_several_rates_that_clear_the_largest_gain_is_chosen() -> None:
    answers = grid({IN_FORCE: 0.2, SLOW: 1.5, FAST: 3.0})

    assert chosen(answers, arm(SLOW, FAST)) == FAST
    assert chosen(answers, arm(FAST, SLOW)) == FAST


def test_the_error_is_the_spread_of_one_draw_of_stays_over_every_repeat() -> None:
    answers = grid({IN_FORCE: 0.5, FAST: 1.0})
    sides = [Side(FIRST, name, budget) for name in (IN_FORCE, FAST) for budget in BUDGETS]
    control, candidate = (
        tuple(answers.ranking(side, seed) for side in pair for seed in SEEDS)
        for pair in (sides[: len(BUDGETS)], sides[len(BUDGETS) :])
    )

    [fast] = choose(
        answers, arm(FAST), rate_in_force=0.001, budgets=BUDGETS, bootstrap=BOOTSTRAP
    ).others

    paired = PairedUnitRankings(control=control, candidate=candidate)
    assert fast.standard_error == pytest.approx(stdev(BOOTSTRAP.reductions(paired)))


def test_repeats_over_the_same_stays_do_not_shrink_the_error() -> None:
    once = grid({IN_FORCE: 0.5, FAST: 1.0})
    # The same answers under a second campaign, as two backbones that answered alike would.
    twice = Answers(
        [
            row._replace(campaign=campaign)
            for campaign in (FIRST, SECOND)
            for candidate in (IN_FORCE, FAST)
            for budget in BUDGETS
            for seed in SEEDS
            for row in answers_of(
                FIRST, candidate, budget, seed, signal=0.5 if candidate == IN_FORCE else 1.0
            )
        ]
    )

    [alone] = choose(
        once, arm(FAST), rate_in_force=0.001, budgets=BUDGETS, bootstrap=BOOTSTRAP
    ).others
    [doubled] = choose(
        twice,
        arm(FAST, campaigns=(FIRST, SECOND)),
        rate_in_force=0.001,
        budgets=BUDGETS,
        bootstrap=BOOTSTRAP,
    ).others

    assert doubled.repeats == 2 * alone.repeats
    assert doubled.standard_error == pytest.approx(alone.standard_error)
    assert doubled.gain == pytest.approx(alone.gain)


def test_a_rate_missing_a_side_or_run_under_other_seeds_is_refused() -> None:
    answers = grid({IN_FORCE: 1.0, FAST: 1.0})
    with pytest.raises(ValueError, match="no answer"):
        chosen(answers, arm(SLOW))

    short = Answers(
        row
        for candidate, seeds in ((IN_FORCE, SEEDS), (FAST, SEEDS[:1]))
        for budget in BUDGETS
        for seed in seeds
        for row in answers_of(FIRST, candidate, budget, seed, signal=1.0)
    )
    with pytest.raises(ValueError, match="ran under seeds"):
        chosen(short, arm(FAST))


def test_the_choices_are_written_one_row_per_rate(tmp_path: Path) -> None:
    choice = choose(
        grid({IN_FORCE: 0.5, SLOW: 0.5, FAST: 2.5}),
        arm(SLOW, FAST),
        rate_in_force=0.001,
        budgets=BUDGETS,
        bootstrap=BOOTSTRAP,
    )

    write_choices(tmp_path / CHOICES, [choice])

    with (tmp_path / CHOICES).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["candidate"] for row in rows] == [IN_FORCE, SLOW, FAST]
    assert [row["chosen"] for row in rows] == ["False", "False", "True"]
    assert float(rows[0]["rate"]) == 0.001


def test_an_arm_needs_its_campaigns_the_rate_in_force_and_another_rate() -> None:
    with pytest.raises(SystemExit):
        parse_arguments(
            ["--predictions", "d", "--rate-in-force", "0.001", "--arm", "a", FIRST, IN_FORCE]
        )
    arguments = parse_arguments(
        [
            "--predictions",
            "d",
            "--rate-in-force",
            "0.001",
            "--arm",
            "fine-tuning",
            f"{FIRST}+{SECOND}",
            IN_FORCE,
            SLOW,
            FAST,
        ]
    )

    assert arms_of(arguments) == [
        Arm(
            label="fine-tuning",
            campaigns=(FIRST, SECOND),
            in_force=IN_FORCE,
            others=(SLOW, FAST),
        )
    ]
    assert arguments.budgets == ["20", "50", "200"]
