"""Fine-tuning's gain on the stays a backbone read beside its gain on the stays it left."""

import csv
import random
from pathlib import Path

import pytest

from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from scripts.campaign_pairs_report import Answers, PredictionRow
from scripts.transduction_report import (
    MEAN,
    TRANSDUCTION,
    BudgetReading,
    Pair,
    parse_arguments,
    render,
    transduction,
    write_transduction,
)

FIRST = "596849cd-fea8-4520-800f-be0f0e117ee0"
SECOND = "0af04f55-5c2e-4aa9-8d60-c8ca6e52b088"
BACKBONE = "ef2674dd-b0af-4a84-9f3e-49b8a1126e13"
SCRATCH = "from_scratch"
TUNED = "full_fine_tuning"
SEEDS = (1, 2)
UNITS = 420
BOOTSTRAP = PairedUnitBootstrap(resamples=200, seed=1)
# Every other stay was read in pretraining.
SEEN = frozenset(f"set-a/{index:05d}" for index in range(0, UNITS, 2))


def answers_of(
    campaign: str, candidate: str, budget: str, seed: int, signal: float, lift_on_seen: float
) -> list[PredictionRow]:
    """One repeat: every seventh stay dies, and the answer leans towards it by ``signal``."""
    draws = random.Random(f"{campaign}/{candidate}/{budget}/{seed}")
    rows = []
    for index in range(UNITS):
        unit = f"set-a/{index:05d}"
        dies = index % 7 == 0
        lean = signal + (lift_on_seen if unit in SEEN else 0.0)
        rows.append(
            PredictionRow(
                campaign=campaign,
                candidate=candidate,
                budget=budget,
                seed=seed,
                unit=unit,
                position=0,
                ends_at=48.0,
                target=1.0 if dies else 0.0,
                predicted=draws.gauss(lean if dies else 0.0, 1.0),
            )
        )
    return rows


def campaign_rows(
    campaign: str, *, tuned: float, lift_on_seen: float, seeds: tuple[int, ...] = SEEDS
) -> list[PredictionRow]:
    return [
        row for seed in SEEDS for row in answers_of(campaign, SCRATCH, "200", seed, 0.5, 0.0)
    ] + [
        row
        for seed in seeds
        for row in answers_of(campaign, TUNED, "200", seed, tuned, lift_on_seen)
    ]


def read(answers: Answers, pairs: list[Pair]) -> list[BudgetReading]:
    return transduction(
        answers, pairs, control=SCRATCH, candidate=TUNED, budgets=["200"], bootstrap=BOOTSTRAP
    )


def test_a_gain_made_on_the_stays_read_shows_as_a_gap() -> None:
    answers = Answers(campaign_rows(FIRST, tuned=0.5, lift_on_seen=2.0))

    [reading] = read(answers, [Pair(label="base1", campaign=FIRST, seen=SEEN)])

    [split] = reading.splits
    assert (split.read, split.unread) == (UNITS // 2, UNITS // 2)
    assert split.gain_read > 0.1
    assert split.gain_unread == pytest.approx(0.0, abs=0.06)
    assert reading.gap == pytest.approx(split.gap)
    assert reading.low > 0.0
    assert reading.standard_error > 0.0


def test_a_gain_made_alike_on_every_stay_leaves_no_gap() -> None:
    answers = Answers(campaign_rows(FIRST, tuned=2.0, lift_on_seen=0.0))

    [reading] = read(answers, [Pair(label="base1", campaign=FIRST, seen=SEEN)])

    assert reading.gain_read > 0.1
    assert reading.gain_unread > 0.1
    assert reading.low < 0.0 < reading.high


def test_backbones_that_read_the_same_stays_share_one_draw() -> None:
    rows = campaign_rows(FIRST, tuned=0.5, lift_on_seen=2.0)
    copied = rows + [row._replace(campaign=SECOND) for row in rows]

    [alone] = read(Answers(rows), [Pair(label="a", campaign=FIRST, seen=SEEN)])
    [both] = read(
        Answers(copied),
        [Pair(label="a", campaign=FIRST, seen=SEEN), Pair(label="b", campaign=SECOND, seen=SEEN)],
    )

    assert both.gap == pytest.approx(alone.gap)
    assert both.standard_error == pytest.approx(alone.standard_error)
    assert (both.low, both.high) == pytest.approx((alone.low, alone.high))


def test_a_backbone_that_read_none_or_all_of_the_scored_stays_is_refused() -> None:
    answers = Answers(campaign_rows(FIRST, tuned=0.5, lift_on_seen=0.0))
    everyone = frozenset(f"set-a/{index:05d}" for index in range(UNITS))

    for seen in (frozenset(), everyone):
        with pytest.raises(ValueError, match="needs stays on both sides"):
            read(answers, [Pair(label="a", campaign=FIRST, seen=seen)])


def test_a_missing_side_or_other_seeds_or_other_stays_are_refused() -> None:
    answers = Answers(campaign_rows(FIRST, tuned=0.5, lift_on_seen=0.0))
    with pytest.raises(ValueError, match="no answer"):
        read(answers, [Pair(label="a", campaign=SECOND, seen=SEEN)])

    short = Answers(campaign_rows(FIRST, tuned=0.5, lift_on_seen=0.0, seeds=SEEDS[:1]))
    with pytest.raises(ValueError, match="ran under seeds"):
        read(short, [Pair(label="a", campaign=FIRST, seen=SEEN)])

    fewer = [
        row
        for row in campaign_rows(SECOND, tuned=0.5, lift_on_seen=0.0)
        if row.unit != "set-a/00001"
    ]
    mixed = Answers(campaign_rows(FIRST, tuned=0.5, lift_on_seen=0.0) + fewer)
    with pytest.raises(ValueError, match="different stays"):
        read(
            mixed,
            [
                Pair(label="a", campaign=FIRST, seen=SEEN),
                Pair(label="b", campaign=SECOND, seen=SEEN),
            ],
        )


def test_each_backbone_and_the_mean_are_written_and_rendered(tmp_path: Path) -> None:
    rows = campaign_rows(FIRST, tuned=0.5, lift_on_seen=2.0)
    readings = read(
        Answers(rows + [row._replace(campaign=SECOND) for row in rows]),
        [Pair(label="a", campaign=FIRST, seen=SEEN), Pair(label="b", campaign=SECOND, seen=SEEN)],
    )

    write_transduction(tmp_path / TRANSDUCTION, readings)

    with (tmp_path / TRANSDUCTION).open(newline="") as handle:
        written = list(csv.DictReader(handle))
    assert [row["label"] for row in written] == ["a", "b", MEAN]
    assert float(written[2]["gap"]) == pytest.approx(readings[0].gap)
    rendered = render(readings)
    assert "| 200 | a | 210 / 210 |" in rendered
    assert f"{readings[0].gap:+.4f}" in rendered


def test_a_pair_names_a_backbone_by_its_identifier() -> None:
    common = ["--predictions", "d", "--corpus", "physionet2012", "--control", SCRATCH]
    common += ["--candidate", TUNED, "--pair", "base1", FIRST]
    with pytest.raises(SystemExit):
        parse_arguments([*common, "base1"])

    arguments = parse_arguments([*common, BACKBONE])

    assert arguments.pair == [["base1", FIRST, BACKBONE]]
    assert arguments.budgets == ["20", "50", "200"]


def test_backbones_that_read_different_stays_are_each_split_by_their_own() -> None:
    rows = campaign_rows(FIRST, tuned=0.5, lift_on_seen=2.0)
    copied = rows + [row._replace(campaign=SECOND) for row in rows]
    # The second backbone read the stays the first left: the lift falls on its unread stays.
    other = frozenset(f"set-a/{index:05d}" for index in range(UNITS)) - SEEN

    [alone] = read(Answers(rows), [Pair(label="a", campaign=FIRST, seen=SEEN)])
    [both] = read(
        Answers(copied),
        [Pair(label="a", campaign=FIRST, seen=SEEN), Pair(label="b", campaign=SECOND, seen=other)],
    )

    first, second = both.splits
    assert first.gap == pytest.approx(alone.gap)
    assert second.gap == pytest.approx(-alone.gap)
    # One draw splits both ways, so the mirrored gaps cancel in every resample.
    assert both.gap == pytest.approx(0.0, abs=1e-12)
    assert (both.low, both.high) == pytest.approx((0.0, 0.0), abs=1e-12)
