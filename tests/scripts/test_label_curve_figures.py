"""The label curve drawn from the pairs report's files; nothing is trained or resampled."""

import csv
from math import sqrt
from pathlib import Path
from statistics import stdev

import pytest

pytest.importorskip("matplotlib")

from scripts.campaign_pairs_report import COMPARISON_COLUMNS, ScoreRow, write_scores
from scripts.label_curve_figures import Gain, Line, main, render
from scripts.pretraining_screen_report import read_pairs

TUNED = "full_fine_tuning"
SCRATCH = "from_scratch@width=64"


def score(campaign: str, candidate: str, budget: str, seed: int, value: float) -> ScoreRow:
    return ScoreRow(
        campaign=campaign,
        purpose="selection",
        measure="auroc_shortfall",
        candidate=candidate,
        budget=budget,
        seed=seed,
        units=2000,
        score=value,
        seconds=1.0,
    )


def scores() -> list[ScoreRow]:
    rows = []
    for budget, base in (("20", 0.60), ("200", 0.70)):
        for seed in (1, 2, 3):
            rows.append(score("a", TUNED, budget, seed, base + 0.01 * seed))
            rows.append(score("b", TUNED, budget, seed, base + 0.03 * seed))
            rows.append(score("a", SCRATCH, budget, seed, base))
    rows.append(score("a", "probe", "500", 1, 0.75))
    return rows


def write_comparisons(path: Path) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        for budget, gain in (("200", 0.02), ("20", -0.01)):
            row = dict.fromkeys(COMPARISON_COLUMNS, "0")
            row.update(
                control_campaign="a+b",
                control=SCRATCH,
                candidate_campaign="a+b",
                candidate=TUNED,
                budget=budget,
                seed_reduction_mean=str(gain),
                seed_reduction_se="0.005",
                low=str(gain - 0.01),
                high=str(gain + 0.01),
            )
            writer.writerow(row)
    return path


def test_a_line_averages_its_campaigns_within_a_seed_before_the_seeds() -> None:
    points = Line(label="tuned", campaigns="a+b", candidate=TUNED).points(scores())

    assert [p.budget for p in points] == [20, 200]
    per_seed = [0.60 + 0.02 * seed for seed in (1, 2, 3)]
    assert points[0].mean == pytest.approx(0.64)
    assert points[0].se == pytest.approx(stdev(per_seed) / sqrt(3))
    assert points[0].seeds == 3


def test_a_line_reads_only_its_own_campaigns() -> None:
    points = Line(label="tuned", campaigns="b", candidate=TUNED).points(scores())

    assert points[0].mean == pytest.approx(0.66)


def test_a_line_at_one_seed_has_no_spread() -> None:
    (point,) = Line(label="probe", campaigns="a", candidate="probe").points(scores())

    assert (point.budget, point.se, point.seeds) == (500, 0.0, 1)


def test_a_line_without_scores_is_refused() -> None:
    with pytest.raises(ValueError, match="no score"):
        Line(label="x", campaigns="c", candidate=TUNED).points(scores())


def test_a_budget_that_is_not_a_count_is_refused() -> None:
    rows = [score("a", TUNED, "all", 1, 0.8)]

    with pytest.raises(ValueError, match="not a count"):
        Line(label="x", campaigns="a", candidate=TUNED).points(rows)


def test_a_campaign_that_holds_no_score_is_refused() -> None:
    with pytest.raises(ValueError, match="under c"):
        Line(label="x", campaigns="a+c", candidate=TUNED).points(scores())


def test_campaigns_holding_different_seeds_at_a_budget_are_refused() -> None:
    rows = [r for r in scores() if not (r.campaign == "b" and r.seed == 3)]

    with pytest.raises(ValueError, match="different seeds"):
        Line(label="x", campaigns="a+b", candidate=TUNED).points(rows)


def test_a_budget_held_by_one_campaign_of_the_line_is_read() -> None:
    rows = [r for r in scores() if not (r.campaign == "b" and r.budget == "200")]

    points = Line(label="x", campaigns="a+b", candidate=TUNED).points(rows)

    assert points[1].mean == pytest.approx(0.72)


def test_a_score_that_is_not_an_area_is_refused() -> None:
    rows = [score("a", TUNED, "20", 1, 14.0)._replace(measure="rmse")]

    with pytest.raises(ValueError, match="not an area"):
        Line(label="x", campaigns="a", candidate=TUNED).points(rows)


def test_a_gain_is_read_by_its_sides_in_budget_order(tmp_path: Path) -> None:
    pairs = read_pairs(write_comparisons(tmp_path / "comparisons.csv"))
    gain = Gain(
        label="g",
        control_campaigns="a+b",
        control=SCRATCH,
        candidate_campaigns="a+b",
        candidate=TUNED,
    )

    assert [(p.budget, p.mean) for p in gain.pairs(pairs)] == [("20", -0.01), ("200", 0.02)]


def test_a_gain_the_report_did_not_make_is_refused(tmp_path: Path) -> None:
    pairs = read_pairs(write_comparisons(tmp_path / "comparisons.csv"))
    gain = Gain(
        label="g",
        control_campaigns="a",
        control=SCRATCH,
        candidate_campaigns="a+b",
        candidate=TUNED,
    )

    with pytest.raises(ValueError, match="no comparison"):
        gain.pairs(pairs)


def test_the_figure_is_drawn_and_its_numbers_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_scores(tmp_path / "scores.csv", scores())
    comparisons = write_comparisons(tmp_path / "comparisons.csv")
    figure = tmp_path / "out" / "curve.png"

    code = main(
        [
            "--figure",
            str(figure),
            "--scores",
            str(tmp_path / "scores.csv"),
            "--comparisons",
            str(comparisons),
            "--line",
            "tuned",
            "a+b",
            TUNED,
            "--line",
            "probe",
            "a",
            "probe",
            "--gain",
            "gain",
            "a+b",
            SCRATCH,
            "a+b",
            TUNED,
            "--units",
            "stays",
        ]
    )

    assert code == 0
    assert figure.stat().st_size > 0
    printed = capsys.readouterr().out
    assert "| tuned | 20 | 0.6400 |" in printed
    assert "| gain | 200 | +0.0200 | [+0.0100; +0.0300] |" in printed


def test_a_gain_without_comparisons_is_refused(tmp_path: Path) -> None:
    write_scores(tmp_path / "scores.csv", scores())

    with pytest.raises(SystemExit):
        main(
            [
                "--figure",
                str(tmp_path / "curve.png"),
                "--scores",
                str(tmp_path / "scores.csv"),
                "--line",
                "tuned",
                "a+b",
                TUNED,
                "--gain",
                "gain",
                "a+b",
                SCRATCH,
                "a+b",
                TUNED,
            ]
        )


def test_the_table_lists_every_point_and_gain() -> None:
    line = Line(label="tuned", campaigns="a+b", candidate=TUNED)

    table = render([(line, line.points(scores()))], [])

    assert table.count("| tuned |") == 2
