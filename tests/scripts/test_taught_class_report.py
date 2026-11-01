"""The classes a taught cell learnt, drawn again as the harness drew them, and the two halves."""

import csv
from pathlib import Path
from uuid import UUID

import pytest

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.detection.taught_events import TaughtEvents
from scripts.campaign_pairs_report import PredictionRow, write_predictions
from scripts.taught_class_report import (
    OTHER_HALF,
    TAUGHT_CLASSES,
    TAUGHT_HALF,
    FittingEvents,
    events_taught_by,
    main,
)

MISSION = "ESA-Mission1"
TASK = TaskId(UUID(int=7))
TAUGHT, UNTAUGHT = (
    "frozen_knn@events=1,reference=unit",
    "frozen_knn@distance=whitened,reference=unit",
)
# Two scored events in the fitting month, of two classes, and one of each class in the scored
# month; the channels are ones the corpus reads.
LABELS = "\n".join(
    [
        "ID,Channel,StartTime,EndTime",
        "id_1,channel_41,2000-01-05T10:00:00.000Z,2000-01-05T12:00:00.000Z",
        "id_2,channel_42,2000-01-20T06:15:00.000Z,2000-01-20T06:45:00.000Z",
        "id_3,channel_41,2000-02-03T10:00:00.000Z,2000-02-03T11:00:00.000Z",
        "id_4,channel_42,2000-02-10T10:00:00.000Z,2000-02-10T11:00:00.000Z",
        "",
    ]
)
TYPES = "\n".join(
    [
        "ID,Class,Subclass,Category,Dimensionality,Locality,Length",
        "id_1,class_a,s,Anomaly,Univariate,Global,Subsequence",
        "id_2,class_b,s,Rare Event,Univariate,Local,Subsequence",
        "id_3,class_a,s,Anomaly,Univariate,Global,Subsequence",
        "id_4,class_b,s,Anomaly,Univariate,Global,Subsequence",
        "",
    ]
)


def raw(tmp_path: Path) -> Path:
    folder = tmp_path / "raw" / MISSION
    folder.mkdir(parents=True)
    (folder / "labels.csv").write_text(LABELS, encoding="utf-8")
    (folder / "anomaly_types.csv").write_text(TYPES, encoding="utf-8")
    return tmp_path / "raw"


def predictions(path: Path) -> Path:
    """Answers of February's windows: the taught cell finds the third event, the untaught none."""
    first_hour = (31 * 24) + 1  # hours since the origin at which February's first window ends
    rows = []
    for candidate in (TAUGHT, UNTAUGHT):
        for position in range(29 * 24):
            hour = float(first_hour + position)
            # The window ending at 02-03T11:00 overlaps event 3, the one at 02-10T11:00 event 4.
            target = 3.0 if position == 2 * 24 + 10 else 4.0 if position == 9 * 24 + 10 else 0.0
            if candidate == TAUGHT and target == 3.0:
                score = 5.0
            elif candidate == UNTAUGHT and target == 3.0:
                score = -1.0
            else:
                score = 0.0 if target else 0.5 - position / 10_000
            rows.append(
                PredictionRow(
                    "c1", candidate, "all", 1, f"{MISSION}/2000-02", position, hour, target, score
                )
            )
    write_predictions(path, rows)
    return path


def test_the_events_a_name_was_taught_are_read_off_it() -> None:
    assert events_taught_by("frozen_knn@events=5,reference=unit") == 5
    assert events_taught_by("summaries_knn@reference=unit,events=10,head=trees") == 10
    assert events_taught_by("frozen_knn@distance=whitened") == 0


def test_the_report_splits_the_scored_events_by_the_classes_the_cell_was_taught(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = raw(tmp_path)
    out = tmp_path / "out"

    main(
        [
            "--task",
            "esa-mission1-events",
            "--task-id",
            str(TASK),
            "--predictions",
            str(predictions(tmp_path / "predictions.csv")),
            "--out",
            str(out),
            "--pair",
            TAUGHT,
            UNTAUGHT,
            "--first",
            "2000-01",
            "--last",
            "2000-01",
            "--raw",
            str(root),
        ]
    )

    rows = list(csv.DictReader((out / TAUGHT_CLASSES).open(newline="")))
    assert [row["half"] for row in rows] == [TAUGHT_HALF, OTHER_HALF]
    taught_class = rows[0]["taught_classes"]
    assert taught_class in {"class_a", "class_b"}
    assert all(row["scored_events"] == "1" for row in rows)
    # The taught cell finds event 3 (class a) first and the untaught finds it last, so the gain
    # falls on the half holding class a, whichever class was taught.
    gains = {row["half"]: float(row["gain"]) for row in rows}
    lifted = TAUGHT_HALF if taught_class == "class_a" else OTHER_HALF
    assert gains[lifted] > 0.5
    assert gains[TAUGHT_HALF if lifted == OTHER_HALF else OTHER_HALF] == pytest.approx(0.0)
    assert "| campaign | candidate | half |" in capsys.readouterr().out


def test_the_fitting_events_are_drawn_as_the_harness_draws_them(tmp_path: Path) -> None:
    from emblema.catalog.adapters.readers.esa_ad_raw import ORIGIN_SECONDS
    from emblema.evaluation.adapters.readers.esa_ad_raw_ground_truth import EsaAdRawGroundTruth

    truth = EsaAdRawGroundTruth(
        raw(tmp_path), channels={MISSION: (41, 42)}, origin_seconds=ORIGIN_SECONDS, window_hours=1.0
    )
    fitting = FittingEvents.of(truth, MISSION, "2000-01", "2000-01")

    assert fitting.classes == {1: "class_a", 2: "class_b", 3: "class_a", 4: "class_b"}
    assert {int(w.target) for w in fitting.pool} == {1, 2}
    drawn = TaughtEvents.drawn(TASK, fitting.pool, events=1, seed=1)
    assert fitting.taught_classes(TASK, 1, 1) == {fitting.classes[drawn.numbers()[0]]}
