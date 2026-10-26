"""The raw audit measures a source from its files as the coverage report measures it by its reader.

Agreement is held on the intensive-care stays, the corpus whose files are already the long table
the audit reads: drawn by the same seed, every unit row the audit writes is the row the report
writes through the reader, on the miniature sample here and on the whole corpus where it was
downloaded. The rest is held to small tables whose answers are known.
"""

import math
from dataclasses import astuple
from pathlib import Path

import pytest

from emblema.catalog.adapters.readers.physionet2012 import Physionet2012CorpusReader
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from scripts.raw_corpora import raw_root
from scripts.raw_regime_audit import LongTable, audit, summary_table
from scripts.regime_coverage_report import Source, UnitRow, measure_source

SAMPLES = Path(__file__).resolve().parents[1] / "data"
STAYS = Source(
    "physionet2012",
    "physionet2012",
    WindowSpec(48.0 + 1.0 / 60.0, 48.0 + 1.0 / 60.0),
    Physionet2012CorpusReader,
)


def stays(root: Path) -> LongTable:
    return LongTable(
        root=root,
        pattern="set-[ab]/*.txt",
        channel="Parameter",
        time="Time",
        clock=True,
        value="Value",
        missing=frozenset({"-1"}),
        skip=frozenset({"RecordID", "Age", "Gender", "Height", "ICUType"}),
    )


def by_unit(rows: list[UnitRow]) -> list[tuple[object, ...]]:
    """The rows in unit order, a measure that is not a number as ``None`` so that it compares."""
    return [
        tuple(None if isinstance(v, float) and math.isnan(v) else v for v in astuple(row))
        for row in sorted(rows, key=lambda row: row.unit)
    ]


def test_the_audit_of_the_sample_stays_is_the_report_through_the_reader() -> None:
    reported, _, report_empty, from_sample = measure_source(
        STAYS, units=10, windows=1, seed=1, sample=True
    )

    audited, empty = audit("physionet2012", stays(SAMPLES / "physionet2012"), units=10, seed=1)

    assert from_sample
    assert len(audited) == 3
    assert by_unit(audited) == by_unit(reported)
    assert empty == report_empty == 1


@pytest.mark.skipif(
    raw_root("physionet2012") is None,
    reason="the challenge's stays are downloaded only on the working machine",
)
def test_the_audit_of_the_downloaded_stays_is_the_report_through_the_reader() -> None:
    root = raw_root("physionet2012")
    assert root is not None
    reported, _, report_empty, _ = measure_source(STAYS, units=300, windows=1, seed=7, sample=False)

    audited, empty = audit("physionet2012", stays(root), units=300, seed=7)

    assert len(audited) + empty == 300
    assert by_unit(audited) == by_unit(reported)
    assert empty == report_empty


def write_table(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def stations(tmp_path: Path) -> LongTable:
    write_table(
        tmp_path / "north" / "readings.csv",
        [
            "site;parameter;time;value",
            "a;nitrate;0;1.0",
            "a;oxygen;0;7.0",
            "a;nitrate;2;1.1",
            "a;oxygen;5;6.0",
            "a;site_id;0;1",
            "b;nitrate;1;0.5",
            "b;nitrate;4;NaN-flag",
            "c;oxygen;3;NaN-flag",
        ],
    )
    return LongTable(
        root=tmp_path,
        pattern="*/*.csv",
        channel="parameter",
        time="time",
        unit="site",
        value="value",
        missing=frozenset({"NaN-flag"}),
        skip=frozenset({"site_id"}),
        delimiter=";",
    )


def test_a_column_names_the_units_and_a_skipped_or_missing_reading_is_none(
    stations: LongTable,
) -> None:
    rows, empty = audit("water", stations, units=10, seed=1)

    [a, b] = sorted(rows, key=lambda row: row.unit)
    assert (a.unit, a.channels, a.observations, a.instants) == ("a", 2, 4, 3)
    assert a.co_observation == pytest.approx((1.0 + 0.5 + 0.5) / 3)
    assert a.rate_spread == pytest.approx(0.39794, abs=1e-5)
    assert (b.unit, b.channels, b.observations, b.channel_share) == ("b", 1, 1, 0.5)
    assert empty == 1


def test_units_are_drawn_by_seed_and_a_rerun_reads_the_same_ones(stations: LongTable) -> None:
    first, _ = audit("water", stations, units=2, seed=1)
    again, _ = audit("water", stations, units=2, seed=1)

    assert by_unit(first) == by_unit(again)
    assert stations.keys() == ["a", "b", "c"]


def test_a_clock_stamp_is_read_in_hours() -> None:
    table = LongTable(root=Path(), pattern="*", channel="c", time="t", clock=True)

    assert table.instant("47:30") == 47.5
    assert table.instant("00:00") == 0.0


def test_a_file_without_a_column_the_table_reads_is_refused(tmp_path: Path) -> None:
    write_table(tmp_path / "x" / "readings.csv", ["unit,parameter,value", "a,b,1"])
    table = LongTable(root=tmp_path, pattern="*/*.csv", channel="parameter", time="time")

    with pytest.raises(ValueError, match="no column time"):
        audit("broken", table, units=1, seed=1)


def test_the_table_gives_the_medians_and_marks_what_was_not_measured(
    stations: LongTable,
) -> None:
    rows, empty = audit("water", stations, units=10, seed=1)

    head, rule, line = summary_table("water", rows, empty)

    assert head.startswith("| Source | Units | Empty | Channels | Readings | Co-observed")
    assert rule.count("---|") == 9
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    assert cells[:3] == ["water", "2", "1"]
    assert cells[6] == "—"
    assert cells[7] == "0.40 [0.40; 0.40]"
    assert summary_table("none", [], 3)[2] == "| none | 0 | 3 | — | — | — | — | — | — |"
