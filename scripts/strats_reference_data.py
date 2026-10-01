"""The intensive-care stays in the form a published network reads, prepared two ways.

STraTS (Tipirneni and Reddy, TKDD 2022) reads a stay as triplets of time, variable and value, as
this project's tokens are, and reaches 0.835 in area from about 3,200 stays where this project's
network from nothing reaches 0.79. Its official code, run on the same stays, separates what the
data hold from what the network makes of them: once on the stays as its own preprocessing
prepares them from the challenge's files, once on this project's tokens of the same stays. The
two differ in the data and nothing else, since the network re-standardises every variable on its
learning side and a standardised token is an affine image of the reading.

Both files hold the stays of the registered task's tuning side. The stays held out by seed 101,
one in five, are scored, as the network's campaigns score them; the rest are divided four to one
into the stays learnt from and the stays the network's early stop reads. The validation side and
the challenge's test set are not read.

Beside the two files goes the difference of their contents, variable by variable: which readings
one holds and the other does not, and how far the values of the readings both hold lie apart once
the tokens are put back in the files' units.

    uv run scripts/strats_reference_data.py --manifest KEY CHECKSUM --out DIR
        [--expect-scored PREDICTIONS_CSV]
"""

import argparse
import csv
import math
import pickle
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.config.settings import Settings
from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
from emblema.entrypoints.configured import configured_store
from emblema.entrypoints.known_ground_truths import KnownGroundTruths
from emblema.evaluation.adapters.blocks.block_corpus_windows import BlockCorpusWindows
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.application.use_cases.define_downstream_task import DefineDownstreamTask
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.ordering import seeded_rank
from emblema.shared.kernel.tokens import TokenWindow
from scripts.reporting import table

TASK = KnownTasks.PHYSIONET_IN_HOSPITAL_DEATH
# The fifth every campaign of the network's diagnosis was scored on.
SCORED_ONE_IN = 5
DIVISION_SEED = 101
# The share of the learning stays the early stop reads, as the network's own preprocessing divides
# its learning side, and the seed that draws it once for every repeat.
EARLY_STOP_SHARE = 0.2
EARLY_STOP_SEED = 1
MINUTES_PER_HOUR = 60
# The descriptors the network embeds apart from the readings, under the names it gives them.
WARDS: Mapping[str, str] = {
    "ICUType/coronary_care": "ICUType_1",
    "ICUType/cardiac_surgery_recovery": "ICUType_2",
    "ICUType/medical": "ICUType_3",
    "ICUType/surgical": "ICUType_4",
}
# Below this many rows after the record identifier a stay is dropped by the network's own
# preprocessing.
FEWEST_ROWS = 6
PREPARED, TOKENS, DIFFERENCES, SPLIT = (
    "strats-prepared.pkl",
    "tokens.pkl",
    "differences.csv",
    "split.csv",
)


class Row(NamedTuple):
    """One triplet as the network reads it: whose, when, what and how much."""

    stay: str
    minute: int
    variable: str
    value: float


@dataclass(frozen=True)
class Division:
    """Which stays the network learns from, stops early on and is scored on.

    Attributes:
        train: Stays learnt from.
        early_stop: Stays the early stop reads.
        scored: Stays scored.
    """

    train: tuple[str, ...]
    early_stop: tuple[str, ...]
    scored: tuple[str, ...]

    def rows(self) -> Iterable[tuple[str, str]]:
        """Every stay with the side it is on, for a file."""
        for side, stays in (("train", self.train), ("early_stop", self.early_stop)):
            yield from ((stay, side) for stay in stays)
        yield from ((stay, "scored") for stay in self.scored)


def divided(tuning: Iterable[UnitKey]) -> Division:
    """The task's tuning stays on the three sides, the scored fifth as the campaigns hold it out."""
    learnt, scored = InnerHoldout(one_in=SCORED_ONE_IN, division_seed=DIVISION_SEED).divided(
        frozenset(tuning), seed=DIVISION_SEED
    )
    ranked = sorted(
        (str(unit) for unit in learnt),
        key=lambda stay: seeded_rank(EARLY_STOP_SEED, "early-stop", stay),
    )
    held = math.ceil(len(ranked) * EARLY_STOP_SHARE)
    return Division(
        train=tuple(sorted(ranked[held:])),
        early_stop=tuple(sorted(ranked[:held])),
        scored=tuple(sorted(str(unit) for unit in scored)),
    )


def token_rows(
    stay: str,
    window: TokenWindow,
    channels: Sequence[PublishedChannel],
    window_length: float,
) -> list[Row]:
    """A window's tokens as triplets: the minute it was read, the variable, the value as read.

    A descriptor sits at minute zero, as the challenge's files put it; a ward is the code the
    network expects, present or absent. The value is the token's, standardised, since the network
    standardises it again.
    """
    rows = []
    for channel_id, value, time, timeless in zip(
        window.channel_ids, window.values, window.times, window.timeless, strict=True
    ):
        name = channels[channel_id - 1].channel
        if timeless:
            rows.append(Row(stay, 0, WARDS.get(name, name), 1.0 if name in WARDS else value))
        else:
            minute = round(time * window_length * MINUTES_PER_HOUR)
            rows.append(Row(stay, minute, name, value))
    return rows


def prepared_rows(stay: str, lines: Iterable[str]) -> list[Row]:
    """A stay's file as the network's own preprocessing prepares it; empty for a stay it drops.

    Its rules, from its ``preprocess_physionet_2012.py``: the record identifier is not a reading;
    a stay with five rows or fewer beside it is dropped; a negative value marks a missing one;
    the ward code becomes a variable of its own; a repeated row is kept once.
    """
    records = list(csv.reader(lines))[2:]
    named = [record for record in records if record[1]]
    if len(named) < FEWEST_ROWS:
        return []
    rows: dict[Row, None] = {}
    for stamp, variable, text in named:
        value = float(text)
        if value < 0.0:
            continue
        hours, minutes = stamp.split(":")
        minute = int(hours) * MINUTES_PER_HOUR + int(minutes)
        if variable == "ICUType":
            rows[Row(stay, minute, f"ICUType_{int(value)}", 1.0)] = None
        else:
            rows[Row(stay, minute, variable, value)] = None
    return list(rows)


def outcomes(path: Path, prefix: str) -> dict[str, float]:
    """Each stay's in-hospital death from the challenge's outcomes file, keyed as tasks key it."""
    with path.open() as file:
        return {
            f"{prefix}{record['RecordID']}": float(record["In-hospital_death"])
            for record in csv.DictReader(file)
        }


def network_input(
    rows: Iterable[Row], labels: Mapping[str, float], division: Division
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    """What the network's dataset unpickles: triplets, outcomes, and the three sides' stays.

    Its dataset unpacks five items from a list, so the file holds this tuple as a list.
    """
    triplets = pd.DataFrame(list(rows), columns=["ts_id", "minute", "variable", "value"])
    stays = [*division.train, *division.early_stop, *division.scored]
    outcome = pd.DataFrame(
        {"ts_id": stays, "in_hospital_mortality": [int(labels[stay]) for stay in stays]}
    )
    return (
        triplets,
        outcome,
        list(division.train),
        list(division.early_stop),
        list(division.scored),
    )


class Difference(NamedTuple):
    """How the two preparations hold one variable.

    Attributes:
        variable: The variable.
        prepared: Readings the network's own preprocessing holds.
        tokens: Readings this project's tokens hold.
        prepared_only: Readings at a stay and minute the tokens lack.
        tokens_only: Readings at a stay and minute the preprocessing lacks.
        largest_gap: Largest difference of value over the readings both hold, in the files' units.
    """

    variable: str
    prepared: int
    tokens: int
    prepared_only: int
    tokens_only: int
    largest_gap: float


def differences(
    prepared: Iterable[Row], tokens: Iterable[Row], restore: Mapping[str, tuple[float, float]]
) -> list[Difference]:
    """Each variable's readings in one preparation and not the other, and the values' largest gap.

    Readings are matched by stay, minute and value order within that minute. A token is put back
    in the files' units by its channel's mean and standard deviation; a ward and a variable without
    statistics are compared as they stand.
    """
    values: dict[str, dict[tuple[str, int], tuple[list[float], list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: ([], []))
    )
    for side, rows in ((0, prepared), (1, tokens)):
        for row in rows:
            value = row.value
            if side == 1 and row.variable in restore:
                mean, std = restore[row.variable]
                value = value * std + mean
            values[row.variable][(row.stay, row.minute)][side].append(value)
    report = []
    for variable, instants in sorted(values.items()):
        counts = [0, 0, 0, 0]
        gap = 0.0
        for left, right in instants.values():
            counts[0] += len(left)
            counts[1] += len(right)
            counts[2] += max(0, len(left) - len(right))
            counts[3] += max(0, len(right) - len(left))
            for a, b in zip(sorted(left), sorted(right), strict=False):
                gap = max(gap, abs(a - b))
        report.append(Difference(variable, counts[0], counts[1], counts[2], counts[3], gap))
    return report


def ref_of(pair: Sequence[str]) -> ArtifactRef:
    key, checksum = pair
    return ArtifactRef(key, Checksum.parse(checksum))


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--manifest",
        nargs=2,
        metavar=("KEY", "CHECKSUM"),
        required=True,
        help="the corpus the registered task is defined over",
    )
    parser.add_argument("--out", type=Path, required=True, help="directory the files go to")
    parser.add_argument("--workspace", type=Path, default=Path("data/workspace"))
    parser.add_argument("--corpora", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--expect-scored",
        type=Path,
        default=None,
        help="a predictions CSV whose stays the scored fifth must be",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    store = configured_store(Settings())
    manifest = ref_of(arguments.manifest)
    blocks = PublishedCorpusBlocks(store, arguments.workspace)
    corpus = BlockCorpusWindows(blocks)
    tasks = InMemoryDownstreamTaskRepository()
    task = tasks.get(
        DefineDownstreamTask(tasks, corpus, SequentialIdGenerator())(
            TASK.defined_over(manifest, corpus.describe(manifest))
        )
    )
    division = divided(task.tuning_units)
    every = frozenset(division.train + division.early_stop + division.scored)
    if arguments.expect_scored is not None:
        with arguments.expect_scored.open() as file:
            expected = {record["unit"] for record in csv.DictReader(file)}
        if expected != set(division.scored):
            sys.exit(f"the scored fifth is not the {len(expected)} stays the campaigns scored")
    if any(not stay.startswith("set-a/") for stay in every):
        sys.exit("a stay outside set A reached the division")

    placed = corpus.windows_of(manifest, [UnitKey(stay) for stay in sorted(every)])
    labelled = task.labelled(
        placed, KnownGroundTruths.under(arguments.corpora).truths_of(task.corpus, placed)
    )
    published = blocks.manifest_of(manifest)
    windows = blocks.block_of(published).at([item.window.position for item in labelled])
    tokens = [
        row
        for item, window in zip(labelled, windows, strict=True)
        for row in token_rows(
            str(item.window.unit), window, published.channels, published.window_length
        )
    ]
    token_labels = {str(item.window.unit): item.target for item in labelled}

    raw = arguments.corpora / "physionet2012"
    labels = outcomes(raw / "Outcomes-a.txt", "set-a/")
    prepared = []
    for stay in sorted(every):
        with (raw / f"{stay}.txt").open() as file:
            prepared.extend(prepared_rows(stay, file))
    if any(labels[stay] != target for stay, target in token_labels.items()):
        sys.exit("the task's labels and the outcomes file disagree")

    arguments.out.mkdir(parents=True, exist_ok=True)
    for name, rows in ((PREPARED, prepared), (TOKENS, tokens)):
        with (arguments.out / name).open("wb") as file:
            pickle.dump(list(network_input(rows, labels, division)), file)
    with (arguments.out / SPLIT).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["stay", "side"])
        writer.writerows(division.rows())

    restore = {
        channel.channel: (channel.statistics.mean, channel.statistics.std)
        for channel in published.channels
        if channel.statistics is not None and channel.channel not in WARDS
    }
    report = differences(prepared, tokens, restore)
    with (arguments.out / DIFFERENCES).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(Difference._fields)
        writer.writerows(report)
    print(
        f"{len(division.train)} learnt from, {len(division.early_stop)} for the early stop, "
        f"{len(division.scored)} scored; stays dropped by the preprocessing: "
        f"{len(every) - len({row.stay for row in prepared})}; without a window: "
        f"{len(every) - len(token_labels)}"
    )
    print(
        table(
            ["variable", "prepared", "tokens", "prepared only", "tokens only", "largest gap"],
            [
                [
                    d.variable,
                    str(d.prepared),
                    str(d.tokens),
                    str(d.prepared_only),
                    str(d.tokens_only),
                    f"{d.largest_gap:.3g}",
                ]
                for d in report
            ],
        )
    )


if __name__ == "__main__":
    main()
