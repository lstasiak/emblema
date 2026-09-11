import tomllib
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from scripts import corpus_facts
from scripts.corpus_budget_report import (
    DEFAULT_CONFIG,
    Budget,
    Corpus,
    Measured,
    MeasuredWindow,
    Subsampling,
    Window,
)
from scripts.corpus_facts import (
    decimate,
    measure_cmapss,
    measure_esa_ad,
    measure_physionet2012,
    measure_skab,
    measure_smd,
    read_channel_table,
    regular_windows,
    render,
    select_channels,
    timed_windows,
)

W50S5 = Window(name="w50s5", unit="samples", length=50, stride=5, default=True)
W2S1 = Window(name="w2s1", unit="samples", length=2, stride=1, default=True)
H24S12 = Window(name="h24s12", unit="hours", length=24, stride=12, default=True)
TODAY = date(2026, 9, 12)


def corpus(key: str, *windows: Window) -> Corpus:
    """The shipped corpus definition with the window variants a tiny fixture can exercise."""
    return Budget.load(DEFAULT_CONFIG).corpora[key].model_copy(update={"windows": windows})


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --- shared counting helpers --------------------------------------------------------------------


def test_regular_windows_sum_full_windows_over_units_and_multiply_by_channels():
    (measured,) = regular_windows([49, 50, 55], channels=2, windows=[W50S5])

    assert (measured.count, measured.tokens) == (0 + 1 + 2, 3 * 50 * 2)


def test_regular_windows_reject_a_time_based_variant():
    with pytest.raises(SystemExit, match="windowed in samples"):
        regular_windows([100], channels=1, windows=[H24S12])


def test_timed_windows_count_observations_inside_each_window():
    # Windows over 48 h at 24 h / 12 h: [0, 24), [12, 36), [24, 48).
    (measured,) = timed_windows([[0.5, 1.5, 25.0]], [48.0], [H24S12])

    assert (measured.count, measured.tokens) == (3, 2 + 1 + 1)


def test_timed_windows_include_the_start_and_exclude_the_end_of_a_window():
    (measured,) = timed_windows([[0.0, 24.0]], [48.0], [H24S12])

    assert (measured.count, measured.tokens) == (3, 1 + 1 + 1)


def test_timed_windows_skip_windows_that_fall_into_a_gap():
    (measured,) = timed_windows([[0.5, 1.5]], [48.0], [H24S12])

    assert (measured.count, measured.tokens) == (1, 2)


def test_timed_windows_reject_a_sample_based_variant():
    with pytest.raises(SystemExit, match="windowed in hours"):
        timed_windows([[1.0]], [48.0], [W50S5])


# --- measurers on synthetic raw files -----------------------------------------------------------


def cmapss_rows(engine: int, cycles: int, columns: int = 26) -> str:
    return "".join(
        f"{engine} {cycle} " + " ".join("0.0" for _ in range(columns - 2)) + "\n"
        for cycle in range(1, cycles + 1)
    )


def test_cmapss_units_are_engines_and_channels_exclude_settings(tmp_path: Path):
    write(tmp_path / "CMAPSSData" / "train_FD001.txt", cmapss_rows(1, 60) + cmapss_rows(2, 50))
    write(tmp_path / "CMAPSSData" / "test_FD001.txt", cmapss_rows(1, 500))

    measured = measure_cmapss(tmp_path, corpus("cmapss", W50S5), TODAY)

    assert (measured.units, measured.channels, measured.observations) == (2, 21, 110 * 21)
    assert measured.windows == (MeasuredWindow(name="w50s5", count=3 + 1, tokens=4 * 50 * 21),)
    assert "train_FD001: 2 engines" in measured.notes


def test_cmapss_rejects_files_with_different_column_counts(tmp_path: Path):
    write(tmp_path / "train_FD001.txt", cmapss_rows(1, 3))
    write(tmp_path / "train_FD002.txt", cmapss_rows(1, 3, columns=25))

    with pytest.raises(SystemExit, match="inconsistent column counts"):
        measure_cmapss(tmp_path, corpus("cmapss", W50S5), TODAY)


def test_skab_units_are_files_under_data_and_channels_exclude_labels(tmp_path: Path):
    header = "datetime;Accelerometer1RMS;Current;anomaly;changepoint\n"
    rows = "".join(f"2020-03-09 10:14:{38 + i:02d};1.0;2.0;0.0;0.0\n" for i in range(3))
    write(tmp_path / "SKAB-x" / "data" / "valve1" / "0.csv", header + rows)
    write(tmp_path / "SKAB-x" / "data" / "anomaly-free" / "anomaly-free.csv", header + rows)
    write(tmp_path / "SKAB-x" / "docs" / "table.csv", "not;data\n1;2\n")

    measured = measure_skab(tmp_path, corpus("skab", W2S1), TODAY)

    assert (measured.units, measured.channels, measured.observations) == (2, 2, 6 * 2)
    assert measured.windows == (MeasuredWindow(name="w2s1", count=2 * 2, tokens=4 * 2 * 2),)
    assert "median spacing 1 s" in measured.notes


def test_smd_counts_train_files_only(tmp_path: Path):
    train = tmp_path / "OmniAnomaly-x" / "ServerMachineDataset" / "train"
    write(train / "machine-1-1.txt", "0,1,2\n" * 4)
    write(train / "machine-2-1.txt", "0,1,2\n" * 5)
    write(train.with_name("test") / "machine-1-1.txt", "0,1,2\n" * 400)

    measured = measure_smd(tmp_path, corpus("smd", W2S1), TODAY)

    assert (measured.units, measured.channels, measured.observations) == (2, 3, 9 * 3)
    assert measured.windows == (MeasuredWindow(name="w2s1", count=3 + 4, tokens=7 * 2 * 3),)
    assert "2 machines in groups 1: 1, 2: 1" in measured.notes


def test_physionet_descriptors_are_timeless_and_a_second_weight_is_an_observation(tmp_path: Path):
    record = (
        "Time,Parameter,Value\n"
        "00:00,RecordID,132539\n"
        "00:00,Age,54\n"
        "00:00,Weight,80\n"
        "00:00,Weight,81\n"
        "01:00,HR,80\n"
        "30:00,HR,90\n"
    )
    write(tmp_path / "set-a" / "132539.txt", record)
    write(tmp_path / "set-b" / "142675.txt", record)

    measured = measure_physionet2012(tmp_path, corpus("physionet2012", H24S12), TODAY)

    assert (measured.units, measured.channels, measured.observations) == (1, 2, 3)
    assert measured.timeless_tokens_per_unit == 2
    assert measured.windows == (MeasuredWindow(name="h24s12", count=3, tokens=2 + 1 + 1),)
    assert "set-b: 1, set-c: 0" in measured.notes


# --- satellite telemetry helpers ----------------------------------------------------------------


def test_decimate_keeps_one_bin_per_spacing_with_native_counts():
    channel = decimate(np.array([0, 10, 20, 35, 100], dtype=np.int64), spacing=30)

    assert (channel.first, channel.last) == (0, 100)
    assert channel.bins.tolist() == [0, 1, 3]
    assert channel.counts.tolist() == [3, 1, 1]
    assert channel.median_spacing == 12.5


def test_decimate_rejects_an_empty_channel():
    with pytest.raises(SystemExit, match="without observations"):
        decimate(np.array([], dtype=np.int64), spacing=30)


NAMES = ["channel_1", "channel_18", "channel_28", "channel_40"]
TARGETS = [True, False, True, False]
LIGHTWEIGHT = Subsampling(
    channel_set="lightweight", min_spacing_seconds=30, lightweight={"ESA-Mission2": (18, 28)}
)


def test_lightweight_policy_selects_the_declared_channel_range():
    assert select_channels(LIGHTWEIGHT, "ESA-Mission2", NAMES, TARGETS) == [
        "channel_18",
        "channel_28",
    ]


def test_lightweight_policy_falls_back_to_target_channels_for_an_undeclared_mission():
    assert select_channels(LIGHTWEIGHT, "ESA-Mission3", NAMES, TARGETS) == [
        "channel_1",
        "channel_28",
    ]


def test_all_policy_keeps_every_channel():
    policy = Subsampling(channel_set="all", min_spacing_seconds=30)

    assert select_channels(policy, "ESA-Mission1", NAMES, TARGETS) == NAMES


def test_channel_table_reads_names_and_target_flags_by_column_name(tmp_path: Path):
    write(tmp_path / "channels.csv", "Channel,Subsystem,Target\nchannel_1,1,True\nchannel_2,1,0\n")

    assert read_channel_table(tmp_path / "channels.csv") == (
        ["channel_1", "channel_2"],
        [True, False],
    )


def test_channel_table_without_a_target_column_treats_every_channel_as_target(tmp_path: Path):
    write(tmp_path / "channels.csv", "Channel,Group\nchannel_1,a\n")

    assert read_channel_table(tmp_path / "channels.csv") == (["channel_1"], [True])


def test_channel_table_without_a_channel_column_fails_with_the_columns_seen(tmp_path: Path):
    write(tmp_path / "channels.csv", "Name,Group\nx,a\n")

    with pytest.raises(SystemExit, match="no column names the channel"):
        read_channel_table(tmp_path / "channels.csv")


H1S1 = Window(name="h1s1", unit="hours", length=1, stride=1, default=True)


def mission(root: Path, name: str, channels: dict[str, list[int]], targets: str = "1") -> None:
    """One unpacked mission: the channel table, the directory the measurer looks for, no pickles."""
    rows = "".join(f"{channel},{targets}\n" for channel in channels)
    write(root / name / "channels.csv", "Channel,Target\n" + rows)
    (root / name / "channels").mkdir(parents=True, exist_ok=True)


def serve_channels(monkeypatch: pytest.MonkeyPatch, times: dict[str, dict[str, list[int]]]) -> None:
    """Stand in for the pandas pickles: the times of each channel, by mission and channel name."""

    def read(path: Path) -> np.ndarray:
        return np.array(times[path.parent.parent.name][path.stem], dtype=np.int64)

    monkeypatch.setattr(corpus_facts, "channel_seconds", read)


def test_esa_counts_the_selected_channels_of_the_training_half(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Channels 41-46 are Mission1's lightweight set; 50 is outside it. The mission spans two
    # hours, so the training half ends at 3600 s and the 30-second bins end at index 120.
    times = {"channel_41": [0, 10, 3000, 5000, 7200], "channel_45": [1800, 5400], "channel_50": [0]}
    mission(tmp_path, "ESA-Mission1", times)
    serve_channels(monkeypatch, {"ESA-Mission1": times})

    measured = measure_esa_ad(tmp_path, corpus("esa_ad", H1S1), TODAY)

    assert (measured.units, measured.channels) == (1, 2)
    assert (measured.observations, measured.native_observations) == (3, 4)
    assert measured.windows == (MeasuredWindow(name="h1s1", count=1, tokens=3),)
    assert "3 channels, 2 selected (lightweight)" in measured.notes
    assert "native 4 -> 3 at <= 1 per 30 s" in measured.notes


def test_esa_bins_observations_that_arrive_faster_than_the_spacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    times = {"channel_41": [0, 1, 2, 3, 3600]}
    mission(tmp_path, "ESA-Mission1", times)
    serve_channels(monkeypatch, {"ESA-Mission1": times})

    measured = measure_esa_ad(tmp_path, corpus("esa_ad", H1S1), TODAY)

    assert (measured.observations, measured.native_observations) == (1, 4)
    assert "median native spacing 1 s" in measured.notes


def test_a_mission_outside_the_policy_is_measured_but_not_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first = {"channel_41": [0, 3600]}
    third = {"channel_1": [0, 3600], "channel_2": [0, 3600]}
    mission(tmp_path, "ESA-Mission1", first)
    mission(tmp_path, "ESA-Mission3", third)
    serve_channels(monkeypatch, {"ESA-Mission1": first, "ESA-Mission3": third})

    measured = measure_esa_ad(tmp_path, corpus("esa_ad", H1S1), TODAY)

    assert (measured.units, measured.channels, measured.observations) == (1, 1, 1)
    assert "ESA-Mission3 (measured, not counted)" in measured.notes


def test_esa_refuses_a_mission_whose_policy_selects_no_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    times = {"channel_50": [0, 3600]}
    mission(tmp_path, "ESA-Mission1", times)
    serve_channels(monkeypatch, {"ESA-Mission1": times})

    with pytest.raises(SystemExit, match="lightweight policy selects no channel"):
        measure_esa_ad(tmp_path, corpus("esa_ad", H1S1), TODAY)


# --- output -------------------------------------------------------------------------------------


def test_rendered_block_round_trips_through_toml_into_the_budget_model():
    measured = Measured(
        measured_on=TODAY,
        units=2,
        channels=3,
        observations=100,
        native_observations=250,
        archive_bytes=4096,
        timeless_tokens_per_unit=1,
        notes='quoted "notes"; accents: é',
        windows=(MeasuredWindow(name="w2s1", count=7, tokens=42),),
    )

    parsed = tomllib.loads(render("x", measured))

    assert Measured.model_validate(parsed["corpora"]["x"]["measured"]) == measured
