"""Tokenise one window of two corpora the way pretraining does, and store every token as a row.

The figure that explains the representation is drawn from real tokens rather than from a sketch:
an intensive-care stay, whose sensors are read at their own irregular times, and a turbofan
engine, whose sensors are read every cycle. Both go through the corpus's reader and the
tokeniser that publishes a corpus, with statistics fitted on every unit of the part read, as the
sanity report fits them. The numbers describe the data and belong in no result.

    uv sync --all-extras
    uv run scripts/token_view_report.py --out data/report/token-view

Writes ``tokens.csv`` (one row per token, the raw reading beside the token it became) and
``windows.csv`` (one row per window). Without a raw corpus it reads the miniature test sample
and says so. ``token_view_figures.py`` draws the figure from those files.
"""

import argparse
import csv
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.adapters.readers.cmapss import CmapssCorpusReader
from emblema.catalog.adapters.readers.physionet2012 import Physionet2012CorpusReader
from emblema.catalog.adapters.tokenisation.sliding_window import SlidingWindowTokeniser
from emblema.catalog.domain.channels.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.tokenisation.placed_window import PlacedWindow
from emblema.catalog.domain.tokenisation.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.catalog.ports.corpus_reader import CorpusReader
from scripts.raw_corpora import raw_root

SAMPLES = REPO_ROOT / "tests" / "data"
TOKEN_COLUMNS = (
    "corpus",
    "unit",
    "channel",
    "shown",
    "timeless",
    "raw_time",
    "raw_value",
    "value",
    "time",
    "gap",
)
WINDOW_COLUMNS = ("corpus", "unit", "start", "end", "time_unit", "tokens", "channels")


@dataclass(frozen=True)
class Source:
    """One window the figure shows: where it is read from and which of its sensors are drawn.

    A plain record of the script's choices; nothing reads it from outside the process.
    """

    corpus: str
    subset: str
    unit: str
    sample_unit: str
    window: WindowSpec
    last_window: bool
    time_unit: str
    shown: tuple[str, ...]
    reader: Callable[[Path, tuple[str, ...]], CorpusReader]


# The stay is the one the sanity note drew; the sensors run from every few minutes to a few times
# a day. The engine's last window ends at failure, where its sensors drift.
SOURCES = (
    Source(
        corpus="physionet2012",
        subset="set-a",
        unit="set-a/135365",
        sample_unit="set-a/132539",
        window=WindowSpec(48.0, 48.0),
        last_window=False,
        time_unit="hours since admission",
        shown=("HR", "NIMAP", "Urine", "Temp", "Glucose", "Age", "ICUType/medical"),
        reader=Physionet2012CorpusReader,
    ),
    Source(
        corpus="cmapss",
        subset="FD001",
        unit="FD001/1",
        sample_unit="FD001/39",
        window=WindowSpec(50.0, 5.0),
        last_window=True,
        time_unit="engine cycles",
        shown=("T24", "T50", "P30", "NRc"),
        reader=CmapssCorpusReader,
    ),
)


def root_of(corpus: str, sample: bool) -> tuple[Path, bool]:
    """Where the corpus is read from, and whether that is the miniature sample."""
    root = None if sample else raw_root(corpus)
    return (root, False) if root is not None else (SAMPLES / corpus, True)


def placed_window(
    source: Source, root: Path, sample: bool
) -> tuple[PlacedWindow, TokenisationScheme]:
    reader = source.reader(root, (source.subset,))
    units = list(reader.read_units())
    unfitted = TokenisationScheme.for_vocabulary(ChannelVocabulary()).extended_with(
        source.corpus, reader.describe().channel_schema
    )
    observations = chain.from_iterable(reader.read_observations(unit.key) for unit in units)
    statics = chain.from_iterable(unit.static_features for unit in units)
    scheme = SlidingWindowTokeniser().fit(source.corpus, observations, statics, unfitted)
    key = source.sample_unit if sample else source.unit
    unit = next((unit for unit in units if str(unit.key) == key), None)
    if unit is None:
        raise SystemExit(f"{source.corpus} holds no unit {key}")
    placed = list(
        SlidingWindowTokeniser().tokenise(
            source.corpus, unit, reader.read_observations(unit.key), scheme, source.window
        )
    )
    if not placed:
        raise SystemExit(f"unit {key} of {source.corpus} holds no window of {source.window}")
    return (placed[-1] if source.last_window else placed[0]), scheme


def token_rows(
    source: Source, placed: PlacedWindow, scheme: TokenisationScheme
) -> list[dict[str, object]]:
    """Every token of the window, with the reading it was made from put back beside it."""
    rows: list[dict[str, object]] = []
    extent = placed.extent
    for token in placed.window:
        channel = scheme.vocabulary.entry(token.channel_id).channel
        rows.append(
            {
                "corpus": source.corpus,
                "unit": str(placed.unit),
                "channel": channel,
                "shown": channel in source.shown,
                "timeless": token.timeless,
                "raw_time": "" if token.timeless else extent.start + token.time * extent.length,
                "raw_value": scheme.statistics_of(token.channel_id).denormalise(token.value),
                "value": token.value,
                "time": token.time,
                "gap": token.gap,
            }
        )
    return rows


def window_row(
    source: Source, placed: PlacedWindow, rows: Sequence[dict[str, object]]
) -> dict[str, object]:
    return {
        "corpus": source.corpus,
        "unit": str(placed.unit),
        "start": placed.extent.start,
        "end": placed.extent.end,
        "time_unit": source.time_unit,
        "tokens": len(rows),
        "channels": len({row["channel"] for row in rows}),
    }


def write(path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def measure(out: Path, sources: Sequence[Source] = SOURCES, sample: bool = False) -> list[str]:
    """Tokenise one window per source into ``out``; returns a line per source saying what was read.

    ``sample`` reads the miniature test sample even where the raw corpus is on the machine.
    """
    out.mkdir(parents=True, exist_ok=True)
    tokens: list[dict[str, object]] = []
    windows: list[dict[str, object]] = []
    said = []
    for source in sources:
        root, from_sample = root_of(source.corpus, sample)
        placed, scheme = placed_window(source, root, from_sample)
        rows = token_rows(source, placed, scheme)
        tokens.extend(rows)
        windows.append(window_row(source, placed, rows))
        read = "miniature sample" if from_sample else "raw corpus"
        said.append(f"{source.corpus} {placed.unit} {placed.extent}: {len(rows)} tokens ({read})")
    write(out / "tokens.csv", TOKEN_COLUMNS, tokens)
    write(out / "windows.csv", WINDOW_COLUMNS, windows)
    return said


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "report" / "token-view")
    arguments = parser.parse_args()
    for line in measure(arguments.out):
        print(line)


if __name__ == "__main__":
    main()
