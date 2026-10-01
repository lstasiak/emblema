"""The stays as a published network reads them: the division, both preparations, the gaps."""

from pathlib import Path

import pytest

pytest.importorskip("pandas")

from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.shared.kernel.tokens import Token, TokenWindow
from scripts.strats_reference_data import (
    Row,
    differences,
    divided,
    network_input,
    outcomes,
    prepared_rows,
    token_rows,
)

CHANNELS = (
    PublishedChannel(channel_id=1, corpus="physionet2012", channel="Age", timeless=True),
    PublishedChannel(
        channel_id=2, corpus="physionet2012", channel="ICUType/medical", timeless=True
    ),
    PublishedChannel(channel_id=3, corpus="physionet2012", channel="HR"),
)
HEADER = "Time,Parameter,Value\n00:00,RecordID,132539\n"


def stay_file(*rows: str) -> list[str]:
    return [*HEADER.splitlines(keepends=True), *(f"{row}\n" for row in rows)]


def test_the_division_holds_one_stay_in_five_out_and_one_in_five_of_the_rest_for_the_stop() -> None:
    stays = [UnitKey(f"set-a/{number}") for number in range(100)]

    division = divided(stays)

    assert (len(division.train), len(division.early_stop), len(division.scored)) == (64, 16, 20)
    assert set(division.train) | set(division.early_stop) | set(division.scored) == {
        str(stay) for stay in stays
    }
    assert divided(reversed(stays)) == division


def test_a_token_returns_to_the_minute_it_was_read_and_a_ward_to_its_code() -> None:
    window = TokenWindow.of(
        [
            Token(channel_id=1, value=0.5, time=0.0, gap=0.0, timeless=True),
            Token(channel_id=2, value=0.0, time=0.0, gap=0.0, timeless=True),
            Token(channel_id=3, value=-1.25, time=0.5, gap=0.5),
        ]
    )

    rows = token_rows("set-a/1", window, CHANNELS, window_length=48.0)

    assert sorted(rows) == sorted(
        [
            Row("set-a/1", 0, "Age", 0.5),
            Row("set-a/1", 0, "ICUType_3", 1.0),
            Row("set-a/1", 1440, "HR", -1.25),
        ]
    )


def test_the_preprocessing_drops_negatives_and_repeats_and_names_the_ward() -> None:
    rows = prepared_rows(
        "set-a/1",
        stay_file(
            "00:00,Age,54",
            "00:00,ICUType,4",
            "00:00,Height,-1",
            "00:07,HR,70",
            "00:07,HR,70",
            "47:59,Temp,-17.8",
            "48:00,HR,80",
        ),
    )

    assert rows == [
        Row("set-a/1", 0, "Age", 54.0),
        Row("set-a/1", 0, "ICUType_4", 1.0),
        Row("set-a/1", 7, "HR", 70.0),
        Row("set-a/1", 2880, "HR", 80.0),
    ]


def test_a_stay_of_five_rows_or_fewer_is_dropped_before_its_negatives_are() -> None:
    five = stay_file(
        "00:00,Age,54", "00:00,Gender,1", "00:00,Height,-1", "00:01,HR,1", "00:02,HR,2"
    )

    assert prepared_rows("set-a/1", five) == []
    assert len(prepared_rows("set-a/1", [*five, "00:03,HR,3\n"])) == 5


def test_the_outcomes_are_keyed_as_the_task_keys_its_stays(tmp_path: Path) -> None:
    path = tmp_path / "Outcomes-a.txt"
    path.write_text(
        "RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death\n"
        "132539,6,1,5,-1,0\n132540,16,8,8,-1,1\n"
    )

    assert outcomes(path, "set-a/") == {"set-a/132539": 0.0, "set-a/132540": 1.0}


def test_the_network_reads_triplets_outcomes_and_three_sides_in_its_order() -> None:
    division = divided([UnitKey(f"set-a/{number}") for number in range(10)])
    stays = [*division.train, *division.early_stop, *division.scored]
    rows = [Row(stay, 0, "Age", 50.0) for stay in stays]

    triplets, outcome, train, early_stop, scored = network_input(
        rows, {stay: float(index % 2) for index, stay in enumerate(stays)}, division
    )

    assert list(triplets.columns) == ["ts_id", "minute", "variable", "value"]
    assert list(outcome.ts_id) == stays
    assert (train, early_stop, scored) == (
        list(division.train),
        list(division.early_stop),
        list(division.scored),
    )


def test_the_difference_counts_readings_one_side_lacks_and_restores_tokens_to_units() -> None:
    prepared = [Row("s", 0, "HR", 70.0), Row("s", 5, "HR", 80.0), Row("s", 2880, "HR", 90.0)]
    tokens = [Row("s", 0, "HR", 0.0), Row("s", 5, "HR", 1.0), Row("s", 9, "HR", 1.0)]

    (difference,) = differences(prepared, tokens, {"HR": (70.0, 10.0)})

    assert (difference.prepared, difference.tokens) == (3, 3)
    assert (difference.prepared_only, difference.tokens_only) == (1, 1)
    assert difference.largest_gap == pytest.approx(0.0)
