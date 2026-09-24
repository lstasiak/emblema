"""The token view is drawn from real tokens, so the file it is drawn from has to hold them.

The report runs here on the miniature samples of both corpora, into a temporary directory, and
the figure is drawn from what it wrote.
"""

import csv
from pathlib import Path

import pytest

from scripts.token_view_figures import Row, colours, draw, listed, static_label
from scripts.token_view_report import TOKEN_COLUMNS, measure


@pytest.fixture(scope="module")
def stored(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("token-view")
    measure(out, sample=True)
    return out


def rows_of(directory: Path, name: str) -> list[dict[str, str]]:
    with (directory / name).open() as file:
        return list(csv.DictReader(file))


def test_every_token_of_each_window_is_stored_beside_the_reading_it_came_from(
    stored: Path,
) -> None:
    tokens = rows_of(stored, "tokens.csv")
    windows = rows_of(stored, "windows.csv")

    assert {w["corpus"] for w in windows} == {"physionet2012", "cmapss"}
    for window in windows:
        held = [t for t in tokens if t["corpus"] == window["corpus"]]
        assert len(held) == int(window["tokens"])
        assert len({t["channel"] for t in held}) == int(window["channels"])
    assert tuple(tokens[0]) == TOKEN_COLUMNS


def test_a_timed_token_is_placed_inside_its_window_and_a_static_one_has_no_time(
    stored: Path,
) -> None:
    windows = {w["corpus"]: w for w in rows_of(stored, "windows.csv")}
    for token in rows_of(stored, "tokens.csv"):
        window = windows[token["corpus"]]
        if token["timeless"] == "True":
            assert token["raw_time"] == ""
            assert float(token["time"]) == float(token["gap"]) == 0.0
            continue
        assert float(window["start"]) <= float(token["raw_time"]) < float(window["end"])
        assert 0.0 <= float(token["gap"]) <= float(token["time"]) <= 1.0


def test_the_stay_carries_its_static_facts_and_the_engine_carries_none(stored: Path) -> None:
    tokens = rows_of(stored, "tokens.csv")
    statics = {t["corpus"] for t in tokens if t["timeless"] == "True"}

    assert statics == {"physionet2012"}


def test_the_figure_is_drawn_from_the_stored_files(stored: Path, tmp_path: Path) -> None:
    figure = draw(stored, tmp_path / "token-view.png")

    assert figure.stat().st_size > 0


def row(channel: str, *, timeless: bool = False, shown: bool = True, value: float = 1.0) -> Row:
    return Row(channel, shown, timeless, None if timeless else 1.0, value, 0.0, 0.5, 0.1)


def test_each_shown_sensor_keeps_one_colour_and_static_facts_come_last() -> None:
    hue = colours([row("Age", timeless=True), row("HR"), row("Temp"), row("K", shown=False)])

    assert list(hue) == ["HR", "Temp", "Age"]
    assert len(set(hue.values())) == 3


def test_a_ward_is_named_and_a_number_is_shown() -> None:
    assert static_label(row("ICUType/cardiac_surgery_recovery", timeless=True)) == (
        "ICUType = cardiac surgery recovery"
    )
    assert static_label(row("Age", timeless=True, value=83.0)) == "Age = 83"


def test_the_table_lists_static_facts_then_the_first_run_where_most_sensors_take_turns() -> None:
    stay = [row("Age", timeless=True)] + [
        Row(channel, True, False, 1.0, 1.0, 0.0, time, 0.1)
        for channel, time in (("HR", 0.1), ("HR", 0.2), ("HR", 0.3), ("Temp", 0.4), ("K", 0.5))
    ]

    chosen = listed(stay, 2)

    assert [r.channel for r in chosen] == ["Age", "HR", "Temp"]
