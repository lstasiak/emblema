"""Each side's areas over its seeds, the error over its units, and the budgets pooled."""

import csv
import random
import statistics
from pathlib import Path

import pytest

from scripts.campaign_pairs_report import (
    PREDICTIONS,
    Answers,
    PredictionRow,
    Side,
    write_predictions,
)
from scripts.seed_spread_report import SPREADS, Spread, main, pooled, spreads

CAMPAIGN = "596849cd-fea8-4520-800f-be0f0e117ee0"


def answers_of(
    candidate: str, budget: str, seed: int, *, units: int = 200, signal: float = 1.0
) -> list[PredictionRow]:
    """One repeat: every seventh unit dies, and the answer leans towards it by ``signal``."""
    draws = random.Random(f"{candidate}/{budget}/{seed}")
    return [
        PredictionRow(
            campaign=CAMPAIGN,
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


def test_a_side_reads_its_area_under_every_seed_and_their_spread() -> None:
    rows = [row for seed in (1, 2, 3) for row in answers_of("from_scratch", "200", seed)]

    [found] = spreads(Answers(rows), resamples=50, seed=1)

    assert found.seeds == (1, 2, 3)
    expected = [
        Answers(rows).ranking(Side(CAMPAIGN, "from_scratch", "200"), seed).auroc
        for seed in (1, 2, 3)
    ]
    assert found.areas == pytest.approx(expected)
    assert found.deviation == pytest.approx(statistics.stdev(expected))


def test_the_error_over_units_shrinks_as_the_units_grow() -> None:
    small = [row for seed in (1, 2) for row in answers_of("a", "50", seed, units=140)]
    large = [row for seed in (1, 2) for row in answers_of("a", "all", seed, units=1400)]

    [few] = spreads(Answers(small), resamples=200, seed=1)
    [many] = spreads(Answers(large), resamples=200, seed=1)

    assert many.unit_error < few.unit_error / 2


def test_a_side_under_one_seed_has_no_spread_to_read() -> None:
    assert spreads(Answers(answers_of("a", "200", 1)), resamples=10, seed=1) == []


def test_a_shift_every_side_shares_under_a_seed_leaves_no_residual() -> None:
    def side(candidate: str, level: float) -> Spread:
        return Spread(
            side=Side(CAMPAIGN, candidate, "200"),
            seeds=(1, 2, 3),
            areas=(level - 0.05, level + 0.02, level + 0.03),
            unit_error=0.02,
        )

    [budget] = pooled([side("a", 0.70), side("b", 0.75)])

    assert budget.residual == pytest.approx(0.0, abs=1e-12)
    assert budget.typical_deviation == pytest.approx(statistics.stdev((-0.05, 0.02, 0.03)))


def test_sides_under_different_seeds_leave_the_residual_unread() -> None:
    first = Spread(side=Side(CAMPAIGN, "a", "50"), seeds=(1, 2), areas=(0.6, 0.7), unit_error=0.03)
    second = Spread(side=Side(CAMPAIGN, "b", "50"), seeds=(1, 3), areas=(0.6, 0.7), unit_error=0.03)

    assert pooled([first, second])[0].residual is None


def test_the_spreads_are_written_beside_the_answers(tmp_path: Path) -> None:
    rows = [row for seed in (1, 2) for row in answers_of("a", "200", seed)]
    write_predictions(tmp_path / PREDICTIONS, rows)

    main(["--out", str(tmp_path), "--resamples", "20"])

    with (tmp_path / SPREADS).open() as stream:
        [written] = list(csv.DictReader(stream))
    assert (written["candidate"], written["budget"], written["seeds"]) == ("a", "200", "1 2")


def test_a_run_without_exported_answers_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="campaign_pairs_report"):
        main(["--out", str(tmp_path)])
