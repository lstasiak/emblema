"""A row placed among the rows of another corpus by cosine, under each reading, and summarised."""

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.channel_identity_report import (
    PAIRS,
    QUERIES,
    RANDOM,
    SUMMARY,
    TOP,
    ChannelTable,
    Counterpart,
    PairResult,
    Placement,
    RowReadings,
    VocabularyRow,
    compare,
    families_of,
    queries_of,
    random_largest,
    read_map,
    read_summary,
    render,
    row_readings,
    summarise,
    swapped,
    write_pairs,
    write_queries,
    write_random,
    write_settings,
    write_summary,
)

WIDTH = 8
CHANNELS = ("c1", "c2", "c3", "c4", "c5", "c6")


def entries_of(*corpora: str) -> tuple[VocabularyRow, ...]:
    return tuple(
        VocabularyRow(channel_id=index + 1, corpus=corpus, channel=channel)
        for index, (corpus, channel) in enumerate(
            (corpus, channel) for corpus in corpora for channel in CHANNELS
        )
    )


def table_of(rows: np.ndarray, *corpora: str) -> ChannelTable:
    return ChannelTable(np.vstack([np.zeros((1, WIDTH)), rows]), entries_of(*corpora))


def pairs_of(corpus: str, counterpart_corpus: str, shift: int = 0) -> list[Counterpart]:
    """Channel ``i`` of ``corpus`` is the quantity of channel ``i + shift`` of the other."""
    return [
        Counterpart(
            corpus=corpus,
            channel=CHANNELS[index],
            counterpart_corpus=counterpart_corpus,
            counterpart_channel=CHANNELS[(index + shift) % len(CHANNELS)],
        )
        for index in range(len(CHANNELS))
    ]


@pytest.fixture
def learnt() -> np.ndarray:
    return np.random.default_rng(7).normal(size=(len(CHANNELS), WIDTH))


def test_a_copied_counterpart_is_found_first_under_every_reading_without_a_start(
    learnt: np.ndarray,
) -> None:
    copies = np.roll(learnt, -2, axis=0)  # channel i of "a" copies channel i + 2 of "b"
    table = table_of(np.vstack([copies, learnt]), "a", "b")

    results = compare(table, pairs_of("a", "b", shift=2))

    assert [result.row.candidates for result in results] == [len(CHANNELS)] * len(CHANNELS)
    for result in results:
        assert set(result.row.readings) == {"raw", "centred"}
        for placement in result.row.readings.values():
            assert placement.rank == 1
            assert placement.cosine == pytest.approx(1.0)
            assert placement.largest == pytest.approx(1.0)
            assert table.entry(placement.nearest).channel == result.pair.counterpart_channel
        assert math.isnan(result.row.shift)
        assert math.isnan(result.row.start_norm)


def test_the_shifted_reading_finds_what_the_row_moved_towards_when_the_row_itself_does_not(
    learnt: np.ndarray,
) -> None:
    started = np.random.default_rng(3).normal(size=(len(CHANNELS), WIDTH)) * 10.0
    moved = started + 0.1 * learnt  # a small step towards channel i of "b", from a large draw
    table = table_of(np.vstack([moved, learnt]), "a", "b")
    start = table_of(np.vstack([started, learnt]), "a", "b")

    results = compare(table, pairs_of("a", "b"), start)

    centred = learnt - learnt.mean(axis=0)
    for index, result in enumerate(results):
        shifted = result.row.readings["shifted"]
        expected = float(
            learnt[index]
            @ centred[index]
            / np.linalg.norm(learnt[index])
            / np.linalg.norm(centred[index])
        )
        assert shifted.cosine == pytest.approx(expected)
        assert result.row.shift == pytest.approx(0.1 * np.linalg.norm(learnt[index]))
        assert result.row.start_norm == pytest.approx(np.linalg.norm(started[index]))
    assert sum(result.row.readings["shifted"].rank == 1 for result in results) > sum(
        result.row.readings["raw"].rank == 1 for result in results
    )


def test_a_pair_within_one_corpus_meets_every_row_of_it_but_its_own(learnt: np.ndarray) -> None:
    table = table_of(learnt, "a")

    [result] = compare(
        table,
        [Counterpart(corpus="a", channel="c1", counterpart_corpus="a", counterpart_channel="c2")],
    )

    assert result.row.candidates == len(CHANNELS) - 1
    assert table.entry(result.row.readings["raw"].nearest).channel != "c1"


def test_a_channel_the_table_lacks_or_names_as_its_own_counterpart_is_refused(
    learnt: np.ndarray,
) -> None:
    table = table_of(np.vstack([learnt, learnt]), "a", "b")

    with pytest.raises(ValueError, match="no row for channel 'c9'"):
        compare(
            table,
            [
                Counterpart(
                    corpus="a", channel="c1", counterpart_corpus="b", counterpart_channel="c9"
                )
            ],
        )
    with pytest.raises(ValueError, match="its own counterpart"):
        compare(
            table,
            [
                Counterpart(
                    corpus="a", channel="c1", counterpart_corpus="a", counterpart_channel="c1"
                )
            ],
        )
    with pytest.raises(ValueError, match="same vocabulary"):
        compare(table, pairs_of("a", "b"), table_of(learnt, "a"))


def test_a_static_feature_is_neither_a_query_nor_a_candidate(learnt: np.ndarray) -> None:
    entries = entries_of("a", "b")
    entries = (*entries[:-1], VocabularyRow(channel_id=12, corpus="b", channel="c6", timeless=True))
    table = ChannelTable(np.vstack([np.zeros((1, WIDTH)), learnt, learnt]), entries)

    assert table.ids_of("b") == [7, 8, 9, 10, 11]
    assert [result.row.candidates for result in compare(table, pairs_of("a", "b")[:1])] == [5]
    with pytest.raises(ValueError, match="static feature"):
        table.id_of("b", "c6")


def test_a_family_pairs_each_channel_with_each_other_of_its_quantity(learnt: np.ndarray) -> None:
    names = ("T2@0kft", "T2@10kft", "T2@20kft", "P2@0kft", "P2@10kft", "alone")
    entries = tuple(
        VocabularyRow(channel_id=index + 1, corpus="c", channel=name)
        for index, name in enumerate(names)
    )
    table = ChannelTable(np.vstack([np.zeros((1, WIDTH)), learnt]), entries)

    pairs = families_of(table, "c", r"(.*)@.*")

    assert len(pairs) == 3 * 2 + 2 * 1
    assert {(p.channel, p.counterpart_channel) for p in pairs if p.note == "P2"} == {
        ("P2@0kft", "P2@10kft"),
        ("P2@10kft", "P2@0kft"),
    }
    assert all(p.corpus == p.counterpart_corpus == "c" for p in pairs)
    with pytest.raises(ValueError, match="captures no group"):
        families_of(table, "c", r".*@.*")
    with pytest.raises(ValueError, match="no family of two"):
        families_of(table, "c", r"(alone)")


def test_a_map_swapped_asks_about_each_counterpart_among_the_querys_corpus() -> None:
    [pair] = swapped(
        [
            Counterpart(
                corpus="a", channel="x", counterpart_corpus="b", counterpart_channel="y", note="n"
            )
        ]
    )

    assert pair == Counterpart(
        corpus="b", channel="y", counterpart_corpus="a", counterpart_channel="x", note="n"
    )


def test_a_row_is_placed_among_at_least_two_candidates(learnt: np.ndarray) -> None:
    table = table_of(learnt, "a")

    with pytest.raises(ValueError, match="at least two candidates"):
        row_readings(table, 1, [2], 2, None)


def test_a_table_extended_by_a_wider_one_keeps_its_rows_and_takes_the_rest(
    learnt: np.ndarray,
) -> None:
    fitted = table_of(learnt * 2.0, "a")
    wider = table_of(np.vstack([learnt, learnt]), "a", "b")

    extended = fitted.extended_by(wider)

    assert extended.entries == wider.entries
    assert np.array_equal(extended.rows[: len(CHANNELS) + 1], fitted.rows)
    assert np.array_equal(extended.rows[len(CHANNELS) + 1 :], learnt)
    with pytest.raises(ValueError, match="reach past"):
        wider.extended_by(fitted)
    with pytest.raises(ValueError, match="does not continue"):
        fitted.extended_by(table_of(np.vstack([learnt, learnt]), "b", "a"))


def test_a_table_whose_rows_and_entries_disagree_is_refused(learnt: np.ndarray) -> None:
    with pytest.raises(ValueError, match="does not index"):
        ChannelTable(learnt, entries_of("a"))


def test_every_row_of_a_corpus_is_listed_with_its_nearest_and_no_rank(learnt: np.ndarray) -> None:
    table = table_of(np.vstack([np.roll(learnt, -1, axis=0), learnt]), "a", "b")

    results = queries_of(table, "a", "b")

    assert [result.channel for result in results] == list(CHANNELS)
    for index, result in enumerate(results):
        placement = result.row.readings["raw"]
        assert placement.rank == 0
        assert math.isnan(placement.cosine)
        assert table.entry(placement.nearest).channel == CHANNELS[(index + 1) % len(CHANNELS)]
        assert placement.largest == pytest.approx(1.0)


def test_rows_drawn_at_random_reach_the_same_largest_cosines_under_the_same_seed(
    learnt: np.ndarray,
) -> None:
    table = table_of(learnt, "b")

    first = random_largest(table, "b", 20, seed=5)
    again = random_largest(table, "b", 20, seed=5)
    other = random_largest(table, "b", 20, seed=6)

    assert first == again
    assert first != other
    assert len(first) == 20
    assert all(-1.0 <= one <= 1.0 for one in first)


def placed_at(rank: int, candidates: int = 33) -> PairResult:
    return PairResult(
        pair=Counterpart(corpus="a", channel="x", counterpart_corpus="b", counterpart_channel="y"),
        row=RowReadings(
            candidates=candidates,
            readings={"raw": Placement(rank=rank, cosine=1.0 / rank, nearest=1, largest=0.9)},
            norm=1.0,
            start_norm=float("nan"),
            shift=float("nan"),
        ),
    )


def test_counterparts_found_first_are_summarised_as_found_with_a_small_p_value() -> None:
    [summary] = summarise([placed_at(1)] * 21, resamples=50, permutations=200, seed=1)

    assert (summary.reading, summary.pairs, summary.candidates) == ("raw", 21, 33)
    assert (summary.found, summary.found_low, summary.found_high) == (1.0, 1.0, 1.0)
    assert summary.median_rank == 1.0
    assert summary.cosine == pytest.approx(1.0)
    assert summary.p_value == pytest.approx(1 / 201)


def test_counterparts_at_random_ranks_are_not_told_from_the_permutation() -> None:
    draws = np.random.default_rng(1)
    results = [placed_at(int(rank)) for rank in draws.integers(1, 34, size=21)]

    [summary] = summarise(results, resamples=50, permutations=200, seed=1)

    assert summary.found <= 0.3
    assert summary.p_value > 0.05
    assert summary.found_low <= summary.found <= summary.found_high


def test_a_reading_some_pair_lacks_is_not_summarised() -> None:
    assert summarise([], resamples=5, permutations=5) == []
    with_shift = PairResult(
        pair=placed_at(1).pair,
        row=RowReadings(
            candidates=33,
            readings={
                "raw": Placement(rank=1, cosine=1.0, nearest=1, largest=1.0),
                "shifted": Placement(rank=2, cosine=0.5, nearest=1, largest=0.6),
            },
            norm=1.0,
            start_norm=1.0,
            shift=0.5,
        ),
    )

    assert [
        one.reading for one in summarise([with_shift, placed_at(1)], resamples=5, permutations=5)
    ] == ["raw"]
    assert [one.reading for one in summarise([with_shift], resamples=5, permutations=5)] == [
        "raw",
        "shifted",
    ]


def test_what_is_written_reads_back_and_renders(learnt: np.ndarray, tmp_path: Path) -> None:
    started = np.random.default_rng(3).normal(size=(len(CHANNELS), WIDTH))
    table = table_of(np.vstack([started + learnt, learnt]), "a", "b")
    start = table_of(np.vstack([started, learnt]), "a", "b")
    pairs = pairs_of("a", "b")
    pairs[0] = Counterpart(**{**pairs[0].__dict__, "note": "doubtful"})
    results = compare(table, pairs, start)
    summaries = summarise(results, resamples=20, permutations=20)

    write_settings(tmp_path, {"backbone": "b", "weights": "0123456789abcdef"})
    write_pairs(tmp_path, table, results)
    write_summary(tmp_path, summaries)
    write_queries(tmp_path, table, queries_of(table, "a", "b", start))
    write_random(tmp_path, random_largest(table, "b", 10, seed=1))

    read = read_summary(tmp_path)
    assert [one.reading for one in read] == [one.reading for one in summaries]
    for stored, summary in zip(read, summaries, strict=True):
        assert stored.found == pytest.approx(summary.found)
        assert stored.p_value == pytest.approx(summary.p_value, rel=1e-5)
        assert stored.cosine == pytest.approx(summary.cosine, rel=1e-5)
    with (tmp_path / PAIRS).open(newline="") as handle:
        records = list(csv.DictReader(handle))
    assert records[0]["note"] == "doubtful"
    assert records[0]["shifted_rank"] == "1"
    with (tmp_path / QUERIES).open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == len(CHANNELS)
    with (tmp_path / RANDOM).open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 10
    rendered = render(tmp_path)
    assert f"Found within {TOP}" in rendered
    assert "a/c1" in rendered
    assert "doubtful" in rendered
    assert "95th percentile" in rendered
    assert (tmp_path / SUMMARY).exists()


def test_a_map_reads_its_note_blank_where_the_column_is_absent(tmp_path: Path) -> None:
    path = tmp_path / "map.csv"
    path.write_text("corpus,channel,counterpart_corpus,counterpart_channel\na,c1,b,c2\n")

    assert read_map(path) == [
        Counterpart(corpus="a", channel="c1", counterpart_corpus="b", counterpart_channel="c2")
    ]
