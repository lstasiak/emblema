"""Which stays the sepsis task reads and freezes, on a corpus small enough to count by hand."""

from pathlib import Path

import pytest

from emblema.catalog.adapters.readers.physionet2019 import Physionet2019CorpusReader
from emblema.shared.kernel.ordering import seeded_rank
from scripts.sepsis_task_units import (
    INELIGIBLE_LISTING,
    TEST_LISTING,
    AssessedStay,
    Standing,
    assessed,
    frozen,
    ineligible,
    main,
)

COLUMNS = len(Physionet2019CorpusReader.HEADER)


def stay(
    root: Path,
    unit: str,
    hours: range,
    *,
    septic_from: int | None = None,
    measured_from: int | None = None,
) -> None:
    """A stay of the challenge's format: a heart rate from ``measured_from``, a label from
    ``septic_from``, every other cell empty and the descriptors repeated on every row."""
    subset, _, name = unit.partition("/")
    (root / subset).mkdir(parents=True, exist_ok=True)
    rows = ["|".join(Physionet2019CorpusReader.HEADER)]
    for hour in hours:
        cells = ["NaN"] * COLUMNS
        if measured_from is None or hour >= measured_from:
            cells[0] = "80"
        cells[34:39] = ["60", "1", "NaN", "NaN", "-1"]
        cells[-2] = str(hour)
        cells[-1] = "1" if septic_from is not None and hour >= septic_from else "0"
        rows.append("|".join(cells))
    (root / subset / f"{name}.psv").write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    stay(tmp_path, "training_setA/p000001", range(1, 40))
    stay(tmp_path, "training_setA/p000002", range(3, 50), septic_from=40)
    stay(tmp_path, "training_setA/p000003", range(1, 20))
    stay(tmp_path, "training_setA/p000004", range(1, 40), septic_from=10)
    stay(tmp_path, "training_setB/p100001", range(1, 40), measured_from=26)
    stay(tmp_path, "training_setB/p100002", range(6, 30))
    return tmp_path


def test_every_stay_stands_where_its_first_day_puts_it(corpus: Path) -> None:
    assert assessed(corpus) == [
        AssessedStay("training_setA/p000001", Standing.READ, 0.0),
        AssessedStay("training_setA/p000002", Standing.READ, 1.0),
        AssessedStay("training_setA/p000003", Standing.SHORT, None),
        AssessedStay("training_setA/p000004", Standing.SEPTIC_WITHIN, None),
        AssessedStay("training_setB/p100001", Standing.EMPTY_FIRST_DAY, None),
        # A stay of exactly a day, from hour 6 to the hour past 29, is read.
        AssessedStay("training_setB/p100002", Standing.READ, 0.0),
    ]


def test_every_stay_the_task_does_not_read_is_named_ineligible(corpus: Path) -> None:
    assert ineligible(assessed(corpus)) == [
        "training_setA/p000003",
        "training_setA/p000004",
        "training_setB/p100001",
    ]


def read(hospital: str, outcome: float, count: int) -> list[AssessedStay]:
    return [
        AssessedStay(f"{hospital}/p{int(outcome)}{index:05d}", Standing.READ, outcome)
        for index in range(count)
    ]


def test_a_fifth_of_each_hospital_and_outcome_is_frozen_rounded_up() -> None:
    stays = [
        *read("training_setA", 0.0, 50),
        *read("training_setA", 1.0, 6),
        *read("training_setB", 0.0, 21),
        *read("training_setB", 1.0, 1),
    ]

    chosen = frozen(stays)

    counts = {
        (hospital, outcome): sum(
            1 for unit in chosen if unit.startswith(f"{hospital}/p{int(outcome)}")
        )
        for hospital in ("training_setA", "training_setB")
        for outcome in (0.0, 1.0)
    }
    assert counts == {
        ("training_setA", 0.0): 10,
        ("training_setA", 1.0): 2,
        ("training_setB", 0.0): 5,
        ("training_setB", 1.0): 1,
    }
    assert chosen == sorted(chosen)


def test_the_frozen_stays_follow_the_seed_not_the_order_they_are_listed_in() -> None:
    stays = read("training_setA", 0.0, 40)

    assert frozen(stays) == frozen(list(reversed(stays)))
    assert frozen(stays, seed=1) != frozen(stays, seed=2)


def test_the_frozen_stays_are_not_the_top_of_the_order_the_corpus_holds_out_by() -> None:
    # The Catalog ranks units by seeded_rank(seed, key); a frozen side from the top of that order
    # would leave the corpus's held-out side the band below it.
    stays = read("training_setA", 0.0, 100)
    corpus_order = sorted((stay.unit for stay in stays), key=lambda unit: seeded_rank(1, unit))

    assert set(frozen(stays, seed=1)) != set(corpus_order[:20])


def test_a_stay_the_task_does_not_read_is_never_frozen() -> None:
    stays = [*read("training_setA", 0.0, 5), AssessedStay("training_setA/x", Standing.SHORT, None)]

    assert "training_setA/x" not in frozen(stays)


def test_freezing_every_stay_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two"):
        frozen(read("training_setA", 0.0, 5), one_in=1)


def test_the_listings_and_the_exclusions_are_written_where_each_reader_looks(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    listings, excluded = tmp_path / "listings", tmp_path / "report" / "exclude-units.txt"

    main([str(corpus), "--exclude-units", str(excluded), "--listings", str(listings)])

    test = (listings / TEST_LISTING).read_text(encoding="utf-8").splitlines()
    assert test == frozen(assessed(corpus))
    assert excluded.read_text(encoding="utf-8").splitlines() == [
        unit.partition("/")[2] for unit in test
    ]
    assert (listings / INELIGIBLE_LISTING).read_text(encoding="utf-8").splitlines() == [
        "training_setA/p000003",
        "training_setA/p000004",
        "training_setB/p100001",
    ]
    assert "| training_setA |" in capsys.readouterr().out
