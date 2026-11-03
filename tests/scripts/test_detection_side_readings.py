"""The readings beside the area, off an export; the arithmetic itself is tested with the ranking."""

import csv
from pathlib import Path

import pytest

from scripts.detection_side_readings import COLUMNS, main, read_labels, readings

HEADER = ("campaign", "candidate", "budget", "seed", "unit", "position", "ends_at", "target")


def row(campaign: str, seed: int, position: int, target: float, predicted: float) -> dict[str, str]:
    return {
        "campaign": campaign,
        "candidate": "knn@a=1,b=2",
        "budget": "all",
        "seed": str(seed),
        "unit": "u",
        "position": str(position),
        "ends_at": str(float(position)),
        "target": str(target),
        "predicted": str(predicted),
    }


def cell(campaign: str, seed: int, event_score: float) -> list[dict[str, str]]:
    return [
        row(campaign, seed, 0, 0.0, 0.5),
        row(campaign, seed, 1, 1.0, event_score),
        row(campaign, seed, 2, 0.0, -0.5),
    ]


def test_a_reading_per_cell_with_its_campaigns_label() -> None:
    rows = [*cell("id-1", 1, 0.9), *cell("id-1", 2, 0.1)]

    found = list(readings(rows, share=0.0, beta=0.5, labels={"id-1": "taught"}))

    assert [(each[0], each[1], each[2], each[3]) for each in found] == [
        ("taught", "knn@a=1,b=2", "all", 1),
        ("taught", "knn@a=1,b=2", "all", 2),
    ]
    # The event first: found with no false alarm. The event behind a normal window: not.
    assert found[0][4] == pytest.approx(1.0)
    assert found[0][5] == pytest.approx(1.0)
    assert found[1][4] == pytest.approx(0.5)
    assert found[1][5] == pytest.approx(0.0)


def test_two_budgets_of_one_seed_are_two_cells_not_one_ranking() -> None:
    other = [{**each, "budget": "200"} for each in cell("id-1", 1, 0.1)]

    found = list(readings([*cell("id-1", 1, 0.9), *other], share=0.0, beta=0.5, labels={}))

    assert [each[2] for each in found] == ["all", "200"]
    assert [each[4] for each in found] == [pytest.approx(1.0), pytest.approx(0.5)]


def test_a_cell_whose_rows_do_not_lie_together_is_refused() -> None:
    first = cell("id-1", 1, 0.9)
    rows = [*first[:2], *cell("id-1", 2, 0.9), *first[1:]]

    with pytest.raises(SystemExit, match="do not lie together"):
        list(readings(rows, share=0.01, beta=0.5, labels={}))


def test_labels_are_read_from_their_second_column_and_comments_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "campaigns.txt"
    path.write_text("# from abc\ntaught id-1 digest\nbars id-2\n")

    assert read_labels(path) == {"id-1": "taught", "id-2": "bars"}
    assert read_labels(None) == {}


def test_the_table_holds_a_row_per_cell(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    predictions = tmp_path / "predictions.csv"
    rows = cell("id-1", 1, 0.9)
    with predictions.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(*HEADER, "predicted"))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "beside.csv"

    main(["--predictions", str(predictions), "--out", str(out)])

    with out.open(newline="") as handle:
        table = list(csv.DictReader(handle))
    assert tuple(table[0]) == COLUMNS
    assert table[0]["campaign"] == "id-1"
    assert "1 cells" in capsys.readouterr().out
