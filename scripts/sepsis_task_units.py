"""Derive which stays the sepsis task reads, and which of them are frozen for the final run.

The task asks at the end of a stay's first day, counted from its first recorded row, whether the
challenge's sepsis label turns later in the stay. A stay answers that question when it lasts a
day, holds a measurement within that day, and its label has not turned inside it. Every other
stay is named ineligible, so that the task's sides hold the stays it reads and nothing else: a
stay too short for a day has no window of that length, and the corpus publishes a window for the
other two that the ground truth refuses.

The frozen side is one in five of the stays the task reads, in each hospital and each outcome
apart, so that its share of each hospital and of the septic stays is exact rather than drawn. It is
ranked on a stream of its own: the Catalog holds out the rest by the same seed over the same keys,
and a frozen side cut from the top of that order would leave the next band of it, depleted of the
stays the task reads, as the validation side. It is cut out of the corpus before publication, so
no backbone pretrains on it, and named in the task's listing, as the challenge of 2012 named set C.

The listings it writes ship with the task register and are the record: a test holds them to this
rule over the downloaded corpus.

    uv run scripts/sepsis_task_units.py data/raw/physionet2019/training
        --exclude-units data/report/l1b/exclude-units.txt
"""

import argparse
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.adapters.readers.physionet2019 import Physionet2019CorpusReader
from emblema.entrypoints.known_ground_truths import KnownGroundTruths
from emblema.evaluation.adapters.readers.physionet2019_ground_truth import (
    Physionet2019GroundTruth,
)
from emblema.evaluation.domain.exceptions import UnlabelledWindowError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.shared.kernel.ordering import seeded_rank
from scripts.reporting import table

# One stay in this many of each hospital and outcome is frozen, rounded up.
ONE_IN = 5
SEED = 1
STREAM = "frozen"
LISTINGS = REPO_ROOT / "src" / "emblema" / "entrypoints" / "cli" / "campaign"
TEST_LISTING = "physionet2019_sepsis_test.txt"
INELIGIBLE_LISTING = "physionet2019_sepsis_ineligible.txt"


class Standing(StrEnum):
    """Where a stay stands with the task.

    Attributes:
        SHORT: Shorter than a day; no window of a day is published, so the task never sees it.
        EMPTY_FIRST_DAY: Nothing measured in the first day, so the earliest published window is a
            later one.
        SEPTIC_WITHIN: The label turns inside the first day.
        READ: The task reads the stay's first day.
    """

    SHORT = "short"
    EMPTY_FIRST_DAY = "empty first day"
    SEPTIC_WITHIN = "septic within the first day"
    READ = "read"


@dataclass(frozen=True)
class AssessedStay:
    """One stay as the rule sees it.

    Attributes:
        unit: The stay's key, ``training_set<X>/p<nnnnnn>``.
        standing: Where it stands with the task.
        outcome: Whether its label turns after the first day; ``None`` unless the task reads it.
    """

    unit: str
    standing: Standing
    outcome: float | None

    @property
    def hospital(self) -> str:
        return self.unit.partition("/")[0]

    @property
    def name(self) -> str:
        """The stay's file name, as the publication's exclusion list names it."""
        return self.unit.partition("/")[2]


def assessed(root: Path, *, hours: float = KnownGroundTruths.FIRST_DAY_HOURS) -> list[AssessedStay]:
    """Every stay of both sets, sorted by key, with where it stands."""
    reader = Physionet2019CorpusReader(root)
    truth = Physionet2019GroundTruth(root, KnownGroundTruths.SEPSIS, hours)
    stays: list[AssessedStay] = []
    for unit in reader.read_units():
        first_day_ends = unit.extent.start + hours
        if unit.extent.end < first_day_ends:
            stays.append(AssessedStay(str(unit.key), Standing.SHORT, None))
            continue
        if not any(
            observation.time < first_day_ends for observation in reader.read_observations(unit.key)
        ):
            stays.append(AssessedStay(str(unit.key), Standing.EMPTY_FIRST_DAY, None))
            continue
        first_day = TaskWindow(unit=UnitKey(str(unit.key)), position=0, ends_at=first_day_ends)
        try:
            outcome = truth.truths_of(KnownGroundTruths.PHYSIONET2019, [first_day])[first_day]
        except UnlabelledWindowError:
            stays.append(AssessedStay(str(unit.key), Standing.SEPTIC_WITHIN, None))
            continue
        stays.append(AssessedStay(str(unit.key), Standing.READ, outcome))
    return sorted(stays, key=lambda stay: stay.unit)


def frozen(stays: Iterable[AssessedStay], *, one_in: int = ONE_IN, seed: int = SEED) -> list[str]:
    """The stays frozen for the final run, sorted by key.

    Raises:
        ValueError: If ``one_in`` is below two, which would leave nothing to tune on.
    """
    if one_in < 2:
        raise ValueError(f"one_in must be at least two, got {one_in}")
    groups: dict[tuple[str, float | None], list[str]] = {}
    for stay in stays:
        if stay.standing is Standing.READ:
            groups.setdefault((stay.hospital, stay.outcome), []).append(stay.unit)
    chosen: list[str] = []
    for members in groups.values():
        ranked = sorted(members, key=lambda unit: (seeded_rank(seed, STREAM, unit), unit))
        chosen.extend(ranked[: -(-len(ranked) // one_in)])
    return sorted(chosen)


def ineligible(stays: Iterable[AssessedStay]) -> list[str]:
    """The stays the task does not read, sorted by key."""
    return sorted(stay.unit for stay in stays if stay.standing is not Standing.READ)


def render(stays: Sequence[AssessedStay], test: Sequence[str]) -> str:
    """What the rule did, per hospital: where the stays stand and what the frozen side holds."""
    frozen_units = set(test)
    standing = Counter((stay.hospital, stay.standing) for stay in stays)
    read = Counter(
        (stay.hospital, stay.outcome, stay.unit in frozen_units)
        for stay in stays
        if stay.standing is Standing.READ
    )
    hospitals = sorted({stay.hospital for stay in stays})
    rows = [
        (
            hospital,
            *(f"{standing[(hospital, kind)]:,}" for kind in Standing),
            *(
                f"{read[(hospital, outcome, side)]:,}"
                for side in (False, True)
                for outcome in (0.0, 1.0)
            ),
        )
        for hospital in hospitals
    ]
    return table(
        (
            "Hospital",
            *(kind.value for kind in Standing),
            "rest, no sepsis",
            "rest, sepsis",
            "frozen, no sepsis",
            "frozen, sepsis",
        ),
        rows,
    )


def write_listing(path: Path, units: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{unit}\n" for unit in units), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory holding the challenge's two sets")
    parser.add_argument(
        "--exclude-units",
        type=Path,
        required=True,
        help="file to write the frozen stays to by file name, for the publication to cut out",
    )
    parser.add_argument(
        "--listings",
        type=Path,
        default=LISTINGS,
        help="directory the task register reads its listings from",
    )
    arguments = parser.parse_args(argv)
    stays = assessed(arguments.root)
    test = frozen(stays)
    write_listing(arguments.listings / TEST_LISTING, test)
    write_listing(arguments.listings / INELIGIBLE_LISTING, ineligible(stays))
    write_listing(arguments.exclude_units, (unit.partition("/")[2] for unit in test))
    print(render(stays, test))


if __name__ == "__main__":
    main()
