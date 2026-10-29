"""The three readers of the telemetry's events, each over files written the way its corpus ships.

What they share is checked here rather than in the contract: a window is placed against the
events by its end and the window length the task's publication fixed, every window of an event
answers with that event's number, and a window over a stretch the task does not score answers
with the excluded mark.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from emblema.evaluation.adapters.readers.esa_ad_raw_ground_truth import EsaAdRawGroundTruth
from emblema.evaluation.adapters.readers.opssat_ad_ground_truth import OpssatAdGroundTruth
from emblema.evaluation.adapters.readers.skab_ground_truth import SkabGroundTruth
from emblema.evaluation.domain.exceptions import (
    UnknownGroundTruthError,
    UnreadableGroundTruthError,
)
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from tests.evaluation.support import window

ORIGIN = 946_684_800
MISSION = "ESA-Mission1"
HOUR = 3600.0


def esa(root: Path, labels: str, types: str) -> EsaAdRawGroundTruth:
    folder = root / MISSION
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "labels.csv").write_text(labels, encoding="utf-8")
    (folder / "anomaly_types.csv").write_text(types, encoding="utf-8")
    return EsaAdRawGroundTruth(
        root, channels={MISSION: (41, 42)}, origin_seconds=ORIGIN, window_hours=1.0
    )


# An anomaly over hours 10-12 on a read channel, a rare event over hour 30 on a read channel, a
# gap over hour 40, and an anomaly on a channel the corpus does not read.
LABELS = "\n".join(
    [
        "ID,Channel,StartTime,EndTime",
        "id_1,channel_41,2000-01-01T10:00:00.000Z,2000-01-01T12:00:00.000Z",
        "id_1,channel_42,2000-01-01T10:30:00.000Z,2000-01-01T11:00:00.000Z",
        "id_2,channel_7,2000-01-01T20:00:00.000Z,2000-01-01T21:00:00.000Z",
        "id_3,channel_42,2000-01-02T06:15:00.000Z,2000-01-02T06:45:00.000Z",
        "id_4,channel_41,2000-01-02T16:00:00.000Z,2000-01-02T17:00:00.000Z",
        "",
    ]
)
TYPES = "\n".join(
    [
        "ID,Class,Subclass,Category,Dimensionality,Locality,Length",
        "id_1,class_1,subclass_1,Anomaly,Multivariate,Global,Subsequence",
        "id_2,class_1,subclass_1,Anomaly,Univariate,Global,Subsequence",
        "id_3,class_2,subclass_1,Rare Event,Univariate,Local,Subsequence",
        "id_4,class_3,subclass_1,Communication Gap,,,",
        "",
    ]
)


def test_the_satellites_events_are_numbered_among_the_read_channels(tmp_path: Path) -> None:
    truth = esa(tmp_path, LABELS, TYPES)
    windows = (
        window(f"{MISSION}/2000-01", 0, 9.0),
        window(f"{MISSION}/2000-01", 1, 11.0),
        window(f"{MISSION}/2000-01", 2, 13.0),
        window(f"{MISSION}/2000-01", 3, 21.0),
        window(f"{MISSION}/2000-01", 4, 31.0),
        window(f"{MISSION}/2000-01", 5, 41.0),
    )

    truths = truth.truths_of("esa_ad_raw", windows)

    # The window ending at 13 starts at 12, when the anomaly has just ended; the one ending at
    # 21 overlaps the event on the unread channel alone; the rare event is the second event
    # that counts; the gap is not scored.
    assert [truths[w] for w in windows] == [0.0, 1.0, 0.0, 0.0, 2.0, AnomalyScheme.EXCLUDED]


def test_a_window_over_a_gap_and_an_event_answers_with_the_event(tmp_path: Path) -> None:
    gap_too = LABELS.replace(
        "id_4,channel_41,2000-01-02T16:00:00.000Z",
        "id_4,channel_41,2000-01-02T06:00:00.000Z",
    ).replace("2000-01-02T17:00:00.000Z", "2000-01-02T07:00:00.000Z")
    truth = esa(tmp_path, gap_too, TYPES)

    assert truth.truths_of("esa_ad_raw", [window(f"{MISSION}/2000-01", 0, 31.0)]) == {
        window(f"{MISSION}/2000-01", 0, 31.0): 2.0
    }


def test_a_month_of_a_mission_not_read_is_refused(tmp_path: Path) -> None:
    truth = esa(tmp_path, LABELS, TYPES)

    with pytest.raises(UnknownGroundTruthError, match="mission"):
        truth.truths_of("esa_ad_raw", [window("ESA-Mission2/2000-01", 0, 11.0)])


@pytest.mark.parametrize(
    ("labels", "types", "reason"),
    [
        ("ID,Channel,StartTime,EndTime\nid_1,,x,y\n", TYPES, "no event"),
        (
            "ID,Channel,StartTime,EndTime\nid_1,channel_41,yesterday,2000-01-01T12:00:00.000Z\n",
            TYPES,
            "line 2",
        ),
        (LABELS, "ID,Class\nid_1,class_1\n", "no event or category"),
        (LABELS, "ID,Class,Subclass,Category\nid_1,c,s,Anomaly\n", "names no category"),
    ],
    ids=["no channel", "no instant", "no category column", "event without a category"],
)
def test_malformed_satellite_files_are_refused(
    tmp_path: Path, labels: str, types: str, reason: str
) -> None:
    truth = esa(tmp_path, labels, types)

    with pytest.raises(UnreadableGroundTruthError, match=reason):
        truth.truths_of("esa_ad_raw", [window(f"{MISSION}/2000-01", 0, 11.0)])


def skab(root: Path) -> SkabGroundTruth:
    header = "datetime;Accelerometer1RMS;anomaly;changepoint"
    first = datetime(2020, 3, 9, 10, 0, 0)
    rows = [
        f"{first + timedelta(seconds=second):%Y-%m-%d %H:%M:%S};0.5;{flag};0.0"
        for second, flag in zip(range(0, 300, 10), [0.0] * 20 + [1.0] * 10, strict=True)
    ]
    for folder, name in (("valve1", "0"), ("other", "3")):
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / f"{name}.csv").write_text("\n".join([header, *rows, ""]), encoding="utf-8")
    (root / "anomaly-free").mkdir()
    (root / "anomaly-free" / "anomaly-free.csv").write_text(
        "\n".join(["datetime;Accelerometer1RMS", "2020-02-08 13:30:47;0.5", ""]), encoding="utf-8"
    )
    return SkabGroundTruth(root, window_seconds=100.0)


def test_the_testbeds_experiments_are_the_events_numbered_in_key_order(tmp_path: Path) -> None:
    truth = skab(tmp_path)
    # Rows are ten seconds apart from second 0; the marked ones run from 200 to 290.
    windows = (
        window("valve1/0", 0, 100.0),
        window("valve1/0", 1, 250.0),
        window("other/3", 2, 300.0),
        window("anomaly-free/anomaly-free", 3, 100.0),
    )

    truths = truth.truths_of("skab", windows)

    assert [truths[w] for w in windows] == [0.0, 2.0, 1.0, 0.0]


def test_an_experiment_the_folder_does_not_hold_is_refused(tmp_path: Path) -> None:
    truth = skab(tmp_path)

    with pytest.raises(UnknownGroundTruthError):
        truth.truths_of("skab", [window("valve1/9", 0, 100.0)])
    with pytest.raises(UnknownGroundTruthError):
        truth.truths_of("skab", [window("0", 0, 100.0)])


def test_a_testbed_file_with_another_mark_is_refused(tmp_path: Path) -> None:
    truth = skab(tmp_path)
    (tmp_path / "valve1" / "0.csv").write_text(
        "datetime;Accelerometer1RMS;anomaly;changepoint\n2020-03-09 10:00:00;0.5;2.0;0.0\n",
        encoding="utf-8",
    )

    with pytest.raises(UnreadableGroundTruthError, match="not 0 or 1"):
        truth.truths_of("skab", [window("valve1/0", 0, 100.0)])


def opssat(root: Path, content: str) -> OpssatAdGroundTruth:
    (root / "segments.csv").write_text(content, encoding="utf-8")
    return OpssatAdGroundTruth(root)


SEGMENTS = "\n".join(
    [
        "channel,timestamp,value,label,sampling,anomaly,segment,train",
        "CADC0872,2022-06-01T23:42:54.000Z,0.1,anomaly,1,1,1,1",
        "CADC0872,2022-06-01T23:42:55.000Z,0.2,anomaly,1,1,1,1",
        "CADC0872,2022-06-02T00:00:00.000Z,0.3,a3,1,0,5,0",
        "",
    ]
)


def test_a_fragment_marked_anomalous_is_an_event_of_its_own_number(tmp_path: Path) -> None:
    truth = opssat(tmp_path, SEGMENTS)
    windows = (window("train/1", 0, 1340.0), window("test/5", 1, 1340.0))

    assert truth.truths_of("opssat_ad", windows) == {windows[0]: 1.0, windows[1]: 0.0}


def test_a_fragment_the_file_does_not_hold_is_refused(tmp_path: Path) -> None:
    truth = opssat(tmp_path, SEGMENTS)

    with pytest.raises(UnknownGroundTruthError):
        truth.truths_of("opssat_ad", [window("train/9", 0, 1340.0)])
    with pytest.raises(UnknownGroundTruthError):
        truth.truths_of("opssat_ad", [window("train/one", 0, 1340.0)])


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (SEGMENTS.replace(",1,1,1,1\n", ",1,2,1,1\n", 1), "not 0 or 1"),
        (SEGMENTS.replace("0.2,anomaly,1,1,1,1", "0.2,anomaly,1,0,1,1"), "both ways"),
        ("channel,timestamp,value,label,sampling,anomaly,segment,train\n", "no fragment"),
        ("channel,value\nCADC0872,0.1\n", "line 2"),
    ],
    ids=["mark", "contradiction", "empty", "columns"],
)
def test_malformed_fragment_files_are_refused(tmp_path: Path, content: str, reason: str) -> None:
    truth = opssat(tmp_path, content)

    with pytest.raises(UnreadableGroundTruthError, match=reason):
        truth.truths_of("opssat_ad", [window("train/1", 0, 1340.0)])
