"""The rule's power over designs of seeds and stays, from exported answers."""

import csv
from itertools import product
from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from scripts.comparison_power_report import (
    CHECK,
    COMPONENTS,
    POWER,
    Answers,
    Components,
    Design,
    Pair,
    PairSpec,
    SideKey,
    StayDraws,
    drawn_side,
    main,
    modelled,
    pairs_of,
    pooled_error,
    read_answers,
    replaces,
    simulate,
)

COLUMNS = ("campaign", "candidate", "budget", "seed", "unit", "position", "ends_at", "target")


def answers(seeds: int = 3, stays: int = 60, signal: float = 1.0, seed: int = 0) -> Answers:
    rng = np.random.default_rng(seed)
    outcome = np.arange(stays) % 5 == 0
    scores = rng.normal(size=(seeds, stays)) + signal * outcome
    return Answers(scores=scores, outcome=outcome, units=tuple(f"u{i}" for i in range(stays)))


def brute_area(scores: np.ndarray, outcome: np.ndarray, weights: np.ndarray) -> float:
    total = 0.0
    for i, j in product(np.flatnonzero(outcome), np.flatnonzero(~outcome)):
        hit = 1.0 if scores[i] > scores[j] else 0.5 if scores[i] == scores[j] else 0.0
        total += weights[i] * weights[j] * hit
    return total / (weights[outcome].sum() * weights[~outcome].sum())


def brute_placements(scores: np.ndarray, outcome: np.ndarray, weights: np.ndarray) -> np.ndarray:
    found = np.zeros(len(scores))
    for i in range(len(scores)):
        others = ~outcome if outcome[i] else outcome
        sign = 1.0 if outcome[i] else -1.0
        hits = [
            weights[j] * (1.0 if sign * (scores[i] - scores[j]) > 0 else 0.5)
            for j in np.flatnonzero(others)
            if sign * (scores[i] - scores[j]) >= 0
        ]
        found[i] = sum(hits) / weights[others].sum()
    return found


def test_drawn_areas_and_placements_match_the_pairwise_count_with_ties() -> None:
    side = answers(seeds=2, stays=40)
    side.scores[:, 3] = side.scores[:, 4]  # a tie across the outcome
    side.scores[:, 0] = side.scores[:, 5]
    rng = np.random.default_rng(1)
    counts = rng.integers(0, 3, size=(4, 40))
    counts[:, 0] = 1
    counts[:, 1] = 1
    read = drawn_side(side, counts)
    for draw, seed in product(range(4), range(2)):
        assert read.areas[draw, seed] == pytest.approx(
            brute_area(side.scores[seed], side.outcome, counts[draw])
        )
    expected = np.mean(
        [brute_placements(side.scores[s], side.outcome, counts[2]) for s in range(2)], axis=0
    )
    held = counts[2] > 0
    assert read.placements[2, held] == pytest.approx(expected[held])


def test_the_placement_error_matches_a_stratified_bootstrap() -> None:
    control, candidate = answers(seed=1, stays=400), answers(seed=2, stays=400, signal=1.5)
    every = np.ones((1, 400), dtype=np.int64)
    error = pooled_error(
        drawn_side(control, every), drawn_side(candidate, every), every, control.outcome
    )[0]
    rng = np.random.default_rng(3)
    deaths, survivors = np.flatnonzero(control.outcome), np.flatnonzero(~control.outcome)
    counts = np.zeros((2_000, 400), dtype=np.int64)
    for row in counts:
        np.add.at(row, rng.choice(deaths, len(deaths)), 1)
        np.add.at(row, rng.choice(survivors, len(survivors)), 1)
    gaps = drawn_side(candidate, counts).areas.mean(axis=1) - (
        drawn_side(control, counts).areas.mean(axis=1)
    )
    assert error == pytest.approx(gaps.std(ddof=1), rel=0.1)


def test_stay_draws_keep_the_size_and_the_prevalence() -> None:
    outcome = np.arange(100) % 4 == 0
    drawn = StayDraws(outcome, stays=60, draws=5, seed=1)
    assert (drawn.counts.sum(axis=1) == 60).all()
    assert (drawn.counts[:, outcome].sum(axis=1) == 15).all()


def test_components_recover_known_parts() -> None:
    rng = np.random.default_rng(4)
    effects = rng.normal(0, 0.03, size=10)
    differences = (
        0.01
        + effects
        + rng.normal(0, 0.02, size=(20_000, 1))
        + rng.normal(0, 0.01, size=(20_000, 10))
    )
    parts = Components.of(differences, gap=0.01)
    assert parts.stays == pytest.approx(0.02, rel=0.05)
    assert parts.interaction == pytest.approx(0.01, rel=0.05)
    assert parts.seed == pytest.approx(float(np.std(effects, ddof=1)), rel=0.05)
    assert parts.interval_error(4) == pytest.approx(sqrt(0.02**2 + 0.01**2 / 4), rel=0.05)
    assert parts.seed_error == pytest.approx(sqrt(0.03**2 + 0.01**2) / sqrt(10), rel=0.2)


@pytest.mark.parametrize(
    ("differences", "error", "threshold", "fires"),
    [
        ([0.05, 0.06, 0.04], 0.01, None, True),
        ([0.05, 0.06, 0.04], 0.03, None, False),  # the interval holds zero
        ([0.30, -0.10, -0.05], 0.01, None, False),  # the seeds disagree
        ([0.05, 0.06, 0.04], 0.01, 0.03, False),  # within twice the spread between backbones
        ([0.05, 0.06, 0.04], 0.01, 0.01, True),  # past it: 2·√(0.0058² + 0.01²) is 0.023
        ([0.03, 0.03, 0.03], 0.01, None, True),  # seeds that do not differ have no condition
        ([0.03, 0.03, 0.03], 0.01, 0.02, False),  # but the spread between backbones still holds
    ],
)
def test_the_rule_needs_every_condition(
    differences: list[float], error: float, threshold: float | None, fires: bool
) -> None:
    assert replaces(np.array([differences]), error, threshold).tolist() == [fires]


def test_the_model_replaces_rarely_without_a_gain_and_always_with_a_large_one() -> None:
    parts = Components(
        seed=0.02, stays=0.01, interaction=0.01, effects=np.zeros(10), gap=0.0, seed_error=0.007
    )
    rng = np.random.default_rng(5)
    assert modelled(parts, 10, 0.0, None, 20_000, rng) < 0.06
    assert modelled(parts, 10, 0.2, None, 2_000, rng) == 1.0
    assert modelled(parts, 30, 0.02, None, 20_000, rng) > modelled(
        parts, 5, 0.02, None, 20_000, rng
    )


def sides(*keys: tuple[str, str, str]) -> dict[SideKey, Answers]:
    return {SideKey(*key): answers(stays=10, seed=i) for i, key in enumerate(keys)}


def test_pairs_are_read_at_every_budget_both_sides_hold_in_the_order_of_their_size() -> None:
    held = sides(
        ("a", "frozen_ridge", "all"),
        ("a", "frozen_ridge", "200"),
        ("a", "frozen_ridge", "1000"),
        ("b", "frozen_ridge", "200"),
        ("b", "frozen_ridge", "all"),
        ("b", "frozen_ridge", "500"),
    )
    spec = PairSpec("B", "b / a", "a", "frozen_ridge", "b", "frozen_ridge")

    found = pairs_of([spec], held)

    assert [(p.kind, p.name, p.budget) for p in found] == [
        ("B", "b / a", "200"),
        ("B", "b / a", "all"),
    ]
    assert found[0].control == SideKey("a", "frozen_ridge", "200")
    assert found[0].candidate == SideKey("b", "frozen_ridge", "200")
    assert pairs_of([spec], held, ["all"]) == found[1:]


def test_a_budget_one_side_was_not_exported_at_is_refused() -> None:
    held = sides(("a", "frozen_ridge", "200"), ("b", "frozen_ridge", "500"))
    spec = PairSpec("B", "b / a", "a", "frozen_ridge", "b", "frozen_ridge")

    with pytest.raises(ValueError, match="not exported at 500"):
        pairs_of([spec], held, ["500"])
    with pytest.raises(ValueError, match="shares no budget"):
        pairs_of([spec], held)


def write_predictions(path: Path, rows: list[tuple[object, ...]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((*COLUMNS, "predicted"))
        writer.writerows(rows)


def rows_of(campaign: str, candidate: str, budget: str, side: Answers) -> list[tuple[object, ...]]:
    return [
        (campaign, candidate, budget, s + 1, f"set-a/{u:04d}", u, 48.0, float(side.outcome[u]), v)
        for s, scores in enumerate(side.scores)
        for u, v in enumerate(scores)
    ]


def test_answers_with_two_windows_of_a_stay_are_refused(tmp_path: Path) -> None:
    rows = rows_of("c", "frozen_ridge", "50", answers(seeds=1, stays=5))
    write_predictions(tmp_path / "p.csv", [*rows, rows[0]])
    with pytest.raises(ValueError, match="more than one window"):
        read_answers([tmp_path / "p.csv"], {"c"})


def test_seeds_that_scored_other_stays_are_refused(tmp_path: Path) -> None:
    rows = rows_of("c", "frozen_ridge", "50", answers(seeds=2, stays=5))
    write_predictions(tmp_path / "p.csv", rows[:-1])
    with pytest.raises(ValueError, match="different stays"):
        read_answers([tmp_path / "p.csv"], {"c"})


def test_a_campaign_is_read_from_the_first_file_that_holds_it(tmp_path: Path) -> None:
    first, second = answers(seeds=1, stays=5, seed=1), answers(seeds=1, stays=5, seed=2)
    write_predictions(tmp_path / "a.csv", rows_of("c", "frozen_ridge", "50", first))
    write_predictions(tmp_path / "b.csv", rows_of("c", "frozen_ridge", "50", second))
    read = read_answers([tmp_path / "a.csv", tmp_path / "b.csv"], {"c"})
    assert read[SideKey("c", "frozen_ridge", "50")].scores == pytest.approx(first.scores)


def test_sides_that_scored_other_stays_are_not_paired() -> None:
    control, candidate = answers(stays=20), answers(stays=20, seed=1)
    moved = Answers(candidate.scores, candidate.outcome, (*candidate.units[1:], "other"))
    keys = SideKey("a", "frozen_ridge", "50"), SideKey("b", "frozen_ridge", "50")
    with pytest.raises(ValueError, match="other stays"):
        simulate(
            {keys[0]: control, keys[1]: moved},
            [Pair("B", "b / a", "50", *keys)],
            draws=4,
            readings=10,
            seed=1,
        )


def test_seeds_that_answered_alike_leave_a_seed_error_of_zero_and_the_rule_still_reads() -> None:
    control, candidate = answers(seeds=1, stays=40), answers(seeds=1, stays=40, seed=1)
    alike = {
        SideKey("a", "frozen_ridge", "all"): Answers(
            np.repeat(control.scores, 3, axis=0), control.outcome, control.units
        ),
        SideKey("b", "frozen_ridge", "all"): Answers(
            np.repeat(candidate.scores, 3, axis=0), candidate.outcome, candidate.units
        ),
    }
    pair = Pair("B", "b / a", "all", *alike)

    components, power = simulate(
        alike, [pair], draws=8, readings=10, seed=1, design=Design(gains=(0.0, 0.5))
    )

    assert components[0].seed == 0.0
    assert components[0].seed_error == 0.0
    assert components[0].stays == 40
    direct = [r.power for r in power if r.source == "direct"]
    assert direct[0] < 1.0
    assert direct[1] == 1.0


def test_a_run_writes_every_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows: list[tuple[object, ...]] = []
    candidates = ("frozen_ridge", "untrained_ridge", "full_fine_tuning", "from_scratch")
    for index, campaign in enumerate(("mix-a", "mix-b")):
        for offset, candidate in enumerate(candidates):
            for budget in ("200", "500"):
                side = answers(stays=80, seed=10 * index + offset)
                rows += rows_of(campaign, candidate, budget, side)
    write_predictions(tmp_path / "p.csv", rows)
    with (tmp_path / "comparisons.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "control_campaign",
                "control",
                "candidate_campaign",
                "candidate",
                "budget",
                "low",
                "high",
                "measure",
            )
        )
        writer.writerow(
            ("mix-a", "frozen_ridge", "mix-b", "frozen_ridge", "200", -0.1, 0.1, "auroc_shortfall")
        )
    out = tmp_path / "out"
    main(
        [
            "--predictions",
            str(tmp_path / "p.csv"),
            "--out",
            str(out),
            *("--pair", "A", "a", "mix-a", "untrained_ridge", "mix-a", "frozen_ridge"),
            *("--pair", "B", "b / a", "mix-a", "frozen_ridge", "mix-b", "frozen_ridge"),
            *("--pair", "C", "a", "mix-a", "from_scratch", "mix-a", "full_fine_tuning"),
            *("--pair", "C0", "b / a", "mix-a", "from_scratch", "mix-b", "from_scratch"),
            *("--seed-gap", "B", "0.004", "--seed-gap", "B", "500=0.006"),
            *("--stays", "40", "80"),
            *("--gains", "0", "0.02"),
            *("--seeds", "5", "20"),
            "--check-predictions",
            str(tmp_path / "p.csv"),
            "--check-comparisons",
            str(tmp_path / "comparisons.csv"),
            "--draws",
            "30",
            "--readings",
            "200",
        ]
    )
    with (out / COMPONENTS).open() as stream:
        components = list(csv.DictReader(stream))
    assert {row["kind"] for row in components} == {"A", "B", "C", "C0"}
    assert {row["budget"] for row in components} == {"200", "500"}
    assert {row["stays"] for row in components} == {"40", "80"}
    assert all(row["seed_error"] for row in components)
    with (out / POWER).open() as stream:
        power = list(csv.DictReader(stream))
    assert {row["source"] for row in power} == {"direct", "measured seeds", "model"}
    assert {row["seeds"] for row in power if row["source"] == "model"} == {"5", "20"}
    thresholds = {(row["budget"], row["threshold"]) for row in power if row["kind"] == "B"}
    assert thresholds == {("200", "0.004"), ("500", "0.004"), ("500", "0.006")}
    assert {row["threshold"] for row in power if row["kind"] == "A"} == {""}
    assert (out / CHECK).exists()
    printed = capsys.readouterr().out
    assert "Kind B at 500 stays, spread between backbones 0.004" in printed
    assert "δ=0.02" in printed


def test_a_check_needs_both_files(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--predictions",
                str(tmp_path / "p.csv"),
                "--out",
                str(tmp_path),
                *("--pair", "B", "b / a", "a", "frozen_ridge", "b", "frozen_ridge"),
                "--check-predictions",
                str(tmp_path / "p.csv"),
            ]
        )
