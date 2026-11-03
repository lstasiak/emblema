"""A plan read across processes against one run of the report over the same pairs."""

import csv
from pathlib import Path

import pytest

from scripts import campaign_pairs_fanout, campaign_pairs_report
from scripts.campaign_pairs_fanout import Reading, merge, read_plan, split_answers

PREDICTION_COLUMNS = (
    "campaign",
    "candidate",
    "budget",
    "seed",
    "unit",
    "position",
    "ends_at",
    "target",
    "predicted",
)
CAMPAIGN = "11111111-1111-1111-1111-111111111111"
A, B, C = "knn@a=1,b=2", "knn@a=2,b=2", "knn@a=3,b=2"


def write_export(directory: Path) -> None:
    """Three detectors of one campaign read by area, over twelve units holding four events."""
    directory.mkdir()
    rows = []
    for offset, candidate in enumerate((A, B, C)):
        for seed in (1, 2):
            for unit in range(12):
                event = unit + 1 if unit < 4 else 0
                score = (unit * 7 + offset * 3 + seed) % 11 / 11.0 + (
                    0.3 * offset if event else 0.0
                )
                rows.append(
                    (CAMPAIGN, candidate, "all", seed, f"u{unit:02d}", 0, 0.0, float(event), score)
                )
    with (directory / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PREDICTION_COLUMNS)
        writer.writerows(rows)
    # The report takes the campaign's measure and purpose from its cells.
    with (directory / "cells.csv").open("w", newline="") as handle:
        cells = csv.DictWriter(handle, fieldnames=campaign_pairs_report.CELL_COLUMNS)
        cells.writeheader()
        for candidate in (A, B, C):
            for seed in (1, 2):
                for unit in range(12):
                    cells.writerow(
                        {
                            "campaign": CAMPAIGN,
                            "purpose": "final_comparison",
                            "tier": "S",
                            "measure": "pr_auc_shortfall",
                            "candidate": candidate,
                            "budget": "all",
                            "seed": seed,
                            "unit": f"u{unit:02d}",
                            "squared_error": 0.0,
                            "windows": 1,
                            "seconds": 1.0,
                        }
                    )


def plan_text() -> str:
    return "".join(
        f"pair\t{CAMPAIGN}\t{control}\tall\t{CAMPAIGN}\t{candidate}\tall\n"
        for control, candidate in ((A, B), (A, C), (B, C))
    )


def test_a_plan_says_which_campaigns_and_candidates_each_reading_needs() -> None:
    plan = "\n".join(
        (
            "# the screen",
            "",
            "pair\tc1+c2\tx\tall\tc3\ty\tall",
            "interaction\tc1\tt\ttc\tu\tuc\tall",
        )
    )

    pair, interaction = read_plan(plan)

    assert pair.arguments == ("--pair", "c1+c2", "x", "all", "c3", "y", "all")
    assert pair.campaigns == {"c1", "c2", "c3"}
    assert pair.candidates == {"x", "y"}
    assert interaction.arguments[0] == "--interaction"
    assert interaction.candidates == {"t", "tc", "u", "uc"}


@pytest.mark.parametrize("line", ["pair\tc\tx", "other\tc\tx\tall\tc\ty\tall"])
def test_a_line_that_is_no_reading_is_refused_by_its_number(line: str) -> None:
    with pytest.raises(SystemExit, match="plan line 2"):
        read_plan(f"# first\n{line}\n")


def test_each_reading_gets_the_answers_of_its_sides_only(tmp_path: Path) -> None:
    source = tmp_path / "predictions.csv"
    source.write_text("campaign,candidate,seed\nc1,x,1\nc1,y,1\nc2,x,1\n")
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    readings = [
        Reading(arguments=(), campaigns=frozenset({"c1"}), candidates=frozenset({"x"})),
        Reading(arguments=(), campaigns=frozenset({"c1", "c2"}), candidates=frozenset({"x", "y"})),
    ]

    split_answers(source, readings, [first, second])

    assert (first / "predictions.csv").read_text().splitlines()[1:] == ["c1,x,1"]
    assert len((second / "predictions.csv").read_text().splitlines()) == 4


def test_tables_merge_with_one_header_and_a_missing_table_is_no_table(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "t.csv").write_text("h\n1\n")
    (second / "t.csv").write_text("h\n2\n")
    target = tmp_path / "merged.csv"

    assert merge("t.csv", [first, second], target) == 2
    assert target.read_text() == "h\n1\n2\n"
    assert merge("other.csv", [first, second], tmp_path / "none.csv") == 0
    assert not (tmp_path / "none.csv").exists()


def test_a_plan_read_across_processes_is_byte_for_byte_one_run_over_the_same_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "export"
    write_export(export)
    one_run = tmp_path / "one-run"
    one_run.mkdir()
    for name in ("cells.csv", "predictions.csv"):
        (one_run / name).write_bytes((export / name).read_bytes())
    arguments: list[str] = ["--out", str(one_run), "--resamples", "50"]
    for line in plan_text().splitlines():
        arguments += ["--pair", *line.split("\t")[1:]]
    campaign_pairs_report.main(arguments)
    plan = tmp_path / "plan.tsv"
    plan.write_text(plan_text())
    out = tmp_path / "fanout"
    out.mkdir()
    reports: list[list[str]] = []

    def run_here(directory: Path, reading: Reading) -> int:
        reports.append(list(reading.arguments))
        campaign_pairs_report.main(
            ["--out", str(directory), "--resamples", "50", *reading.arguments]
        )
        return 0

    monkeypatch.setattr(campaign_pairs_fanout, "run_report", run_here)
    campaign_pairs_fanout.main(
        ["--plan", str(plan), "--export", str(export), "--out", str(out), "--jobs", "1"]
    )

    assert len(reports) == 3
    assert (out / "comparisons.csv").read_bytes() == (one_run / "comparisons.csv").read_bytes()


def test_a_reading_that_fails_leaves_no_merged_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "export"
    write_export(export)
    plan = tmp_path / "plan.tsv"
    plan.write_text(plan_text())
    out = tmp_path / "fanout"
    out.mkdir()
    (out / "comparisons.csv").write_text("from an earlier run\n")
    monkeypatch.setattr(campaign_pairs_fanout, "run_report", lambda _d, _r: 1)

    with pytest.raises(SystemExit, match="readings failed"):
        campaign_pairs_fanout.main(
            ["--plan", str(plan), "--export", str(export), "--out", str(out)]
        )

    assert not (out / "comparisons.csv").exists()


def test_an_empty_plan_is_refused(tmp_path: Path) -> None:
    plan = tmp_path / "plan.tsv"
    plan.write_text("# nothing\n")

    with pytest.raises(SystemExit, match="no reading"):
        campaign_pairs_fanout.main(
            ["--plan", str(plan), "--export", str(tmp_path), "--out", str(tmp_path)]
        )
