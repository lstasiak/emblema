"""What the task's sides come to on the corpus itself, on a machine that downloaded it.

The task inherits the split the corpus was published under, so the size of its sides is a fact
about that corpus and that seed rather than a choice made here — and the interval every result of
this task carries is as wide as the held-out side is small. Pinned because the number is quoted
outside the code, and a silent change to it would silently change what the results mean.
"""

from collections import Counter
from pathlib import Path

import pytest

from emblema.catalog.domain.identifiers import UnitKey as CatalogUnitKey
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.catalog.domain.tokenisation.unit_split import UnitSplit
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from tests.support.corpora import raw_root

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
TASK_SUBSET = "FD001"
# The window and the split the corpus was published under.
WINDOW = WindowSpec(length=50.0, stride=5.0)
VALIDATION_FRACTION, SEED = 0.2, 1

pytestmark = pytest.mark.skipif(
    raw_root() is None, reason="the full C-MAPSS download is not on this machine"
)


def cycles_per_engine(root: Path) -> dict[str, int]:
    counted: dict[str, int] = {}
    for subset in SUBSETS:
        engines: Counter[str] = Counter()
        for line in (root / f"train_{subset}.txt").read_text().splitlines():
            if line.strip():
                engines[f"{subset}/{line.split()[0]}"] += 1
        counted |= engines
    return counted


@pytest.fixture(scope="module")
def corpus() -> dict[str, int]:
    root = raw_root()
    assert root is not None  # the module is skipped without it
    return cycles_per_engine(root)


def windows_of(cycles: int) -> int:
    return sum(1 for _ in WINDOW.windows_over(TimeExtent(1.0, cycles + 1.0)))


def test_the_corpus_holds_out_eighteen_of_the_hundred_engines_of_the_task_subset(
    corpus: dict[str, int],
) -> None:
    split = UnitSplit.by_seed([CatalogUnitKey(key) for key in corpus], VALIDATION_FRACTION, SEED)

    of_subset = [key for key in corpus if key.startswith(f"{TASK_SUBSET}/")]
    held_out = [key for key in of_subset if CatalogUnitKey(key) in split.validation]
    assert (len(of_subset), len(held_out)) == (100, 18)


def test_the_sides_of_the_task_hold_the_windows_the_budgets_are_drawn_from(
    corpus: dict[str, int],
) -> None:
    split = UnitSplit.by_seed([CatalogUnitKey(key) for key in corpus], VALIDATION_FRACTION, SEED)

    tuning = sum(
        windows_of(cycles)
        for key, cycles in corpus.items()
        if key.startswith(f"{TASK_SUBSET}/") and CatalogUnitKey(key) in split.training
    )
    validation = sum(
        windows_of(cycles)
        for key, cycles in corpus.items()
        if key.startswith(f"{TASK_SUBSET}/") and CatalogUnitKey(key) in split.validation
    )
    assert (tuning, validation) == (2651, 535)
