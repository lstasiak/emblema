"""Measure how each pretraining corpus is sampled, and how much of each regime a pretraining reads.

Two questions about the data, not the model. Per corpus, how irregular the sampling is that the
encoder meets: per unit, the share of its part's channels it measures, how many of its channels
are read at one instant, how uneven each channel's spacing is and how far its channels' rates lie
apart; per window, its tokens, the share of channels it holds and how many of those it reads once.
Per pretraining, the share of its windows and of its observed values that each corpus and each
sampling regime takes, from the passes and fractions its experiment file reads them with.

    uv sync --all-extras
    uv run scripts/regime_coverage_report.py measure --out data/report/regime-coverage
    uv run --env-file .env.r2 scripts/regime_coverage_report.py exposure \
        --out data/report/regime-coverage --manifests <file of name, key, checksum> \
        --experiments experiments/backbone-mixed4-m.toml experiments/backbone-scale-1e9-m.toml

``measure`` writes ``units.csv``, ``windows.csv`` and ``corpora.csv``; ``exposure`` reads each
corpus's published manifest from the bucket and writes ``exposure.csv``. Both print markdown.
Without a raw corpus ``measure`` reads the miniature test sample and says so. A unit sample is
drawn by seed, so a rerun reads the same units. The numbers describe the data and belong in no
result.
"""

import argparse
import csv
import math
import statistics
import sys
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.adapters.readers.cmapss import CmapssCorpusReader
from emblema.catalog.adapters.readers.esa_ad import EsaAdCorpusReader
from emblema.catalog.adapters.readers.physionet2012 import Physionet2012CorpusReader
from emblema.catalog.adapters.readers.physionet2019 import Physionet2019CorpusReader
from emblema.catalog.adapters.readers.skab import SkabCorpusReader
from emblema.catalog.adapters.readers.tep import TennesseeEastmanCorpusReader
from emblema.catalog.adapters.readers.utsd import UtsdCorpusReader, UtsdReading
from emblema.catalog.adapters.readers.wqp import WaterQualityPortalCorpusReader
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.catalog.ports.corpus_reader import CorpusReader
from emblema.shared.kernel.ordering import seeded_rank
from emblema.shared.kernel.sampling import SamplingRegime
from scripts.raw_corpora import raw_root

SAMPLES = REPO_ROOT / "tests" / "data"
UTSD_PREFIX = UtsdCorpusReader.CORPUS_PREFIX


@dataclass(frozen=True)
class Source:
    """One corpus as a pretraining reads it: its reader and the windows its publication cut.

    A plain record of the publications' choices (``data/report`` holds the commands that made
    them); nothing reads it from outside the process.
    """

    corpus: str
    raw: str
    window: WindowSpec
    reader: Callable[[Path], CorpusReader]


def _utsd(name: str, length: float, independent: bool = False) -> Source:
    def reader(root: Path) -> CorpusReader:
        return UtsdCorpusReader(
            root,
            UtsdCorpusReader.dataset_named(name),
            UtsdReading(independent_channels=independent),
        )

    return Source(name, "utsd", WindowSpec(length, length), reader)


# Every corpus the mixture of four and the curve over the scale of pretraining read, windowed as
# published. The time-series collection is regular by construction; its datasets are read here
# so that the claim is measured rather than taken from the reader.
SOURCES: tuple[Source, ...] = (
    Source(
        "cmapss",
        "cmapss",
        WindowSpec(50.0, 5.0),
        lambda root: CmapssCorpusReader(root, per_condition=True),
    ),
    Source("skab", "skab", WindowSpec(100.0, 10.0), SkabCorpusReader),
    Source("esa_ad", "esa_ad", WindowSpec(1.0, 1.0), EsaAdCorpusReader),
    Source(
        "physionet2012",
        "physionet2012",
        WindowSpec(48.0 + 1.0 / 60.0, 48.0 + 1.0 / 60.0),
        Physionet2012CorpusReader,
    ),
    Source("physionet2019", "physionet2019", WindowSpec(24.0, 12.0), Physionet2019CorpusReader),
    Source("tep", "tep", WindowSpec(50.0, 50.0), TennesseeEastmanCorpusReader),
    _utsd(f"{UTSD_PREFIX}ERA5_pressure", 510.0),
    _utsd(f"{UTSD_PREFIX}ERA5_surface", 1016.0),
    _utsd(f"{UTSD_PREFIX}Energy_australian_electricity_demand_dataset", 1024.0),
    _utsd(f"{UTSD_PREFIX}Energy_london_smart_meters_dataset_without_missing_values", 1024.0),
    _utsd(f"{UTSD_PREFIX}Environment_AustraliaRainfall", 1023.0, independent=True),
    _utsd(f"{UTSD_PREFIX}Environment_BeijingPM25Quality", 1023.0, independent=True),
    _utsd(f"{UTSD_PREFIX}Environment_BenzeneConcentration", 1024.0, independent=True),
    _utsd(f"{UTSD_PREFIX}Health_AtrialFibrillation", 320.0),
    _utsd(f"{UTSD_PREFIX}Health_BIDMC32HR", 1000.0),
    _utsd(f"{UTSD_PREFIX}Health_IEEEPPG", 1000.0),
    _utsd(f"{UTSD_PREFIX}Health_MotorImagery", 1000.0, independent=True),
    _utsd(f"{UTSD_PREFIX}Health_PigArtPressure", 1000.0),
    _utsd(f"{UTSD_PREFIX}Health_PigCVP", 1000.0),
    _utsd(f"{UTSD_PREFIX}Health_SelfRegulationSCP1", 149.0),
    _utsd(f"{UTSD_PREFIX}Health_SelfRegulationSCP2", 144.0),
    _utsd(f"{UTSD_PREFIX}Health_TDBrain_csv", 853.0, independent=True),
    _utsd(f"{UTSD_PREFIX}IoT_baian", 1024.0, independent=True),
    _utsd(f"{UTSD_PREFIX}Nature_EigenWorms", 169.0),
    _utsd(f"{UTSD_PREFIX}Nature_Phoneme", 1024.0),
    _utsd(f"{UTSD_PREFIX}Nature_StarLightCurves", 1024.0),
    _utsd(f"{UTSD_PREFIX}Nature_Worms", 900.0),
    _utsd(f"{UTSD_PREFIX}Nature_kdd_cup_2018_dataset_without_missing_values", 1024.0),
    _utsd(f"{UTSD_PREFIX}Nature_temperature_rain_dataset_without_missing_values", 725.0),
    _utsd(f"{UTSD_PREFIX}Transport_pedestrian_counts_dataset", 1024.0),
    _utsd(f"{UTSD_PREFIX}Web_kaggle_web_traffic_dataset_without_missing_values", 803.0),
)

# Corpora read in the tasks' regime that no pretraining of the scale curve reads; a location of the
# water quality portal is windowed whole here, since only its unit measures are read.
TASK_REGIME_SOURCES: tuple[Source, ...] = (
    Source("wqp", "wqp", WindowSpec(122_712.0, 122_712.0), WaterQualityPortalCorpusReader),
)

# The regime each reader declares in its description. Stated here so that the exposure of a
# pretraining is read without its corpora on the machine; the test holds it to the readers.
REGIMES: Mapping[str, SamplingRegime] = {
    "cmapss": SamplingRegime.REGULAR,
    "skab": SamplingRegime.REGULAR,
    "smd": SamplingRegime.REGULAR,
    "tep": SamplingRegime.REGULAR,
    "esa_ad": SamplingRegime.IRREGULAR,
    "physionet2012": SamplingRegime.IRREGULAR,
    "physionet2019": SamplingRegime.IRREGULAR,
    "wqp": SamplingRegime.IRREGULAR,
}


def regime_of(corpus: str) -> SamplingRegime:
    if corpus.startswith(UTSD_PREFIX):
        return SamplingRegime.REGULAR
    return REGIMES[corpus]


# --- measure -------------------------------------------------------------------------------------

Times = Mapping[str, NDArray[np.float64]]


@dataclass(frozen=True)
class UnitRow:
    """The sampling of one unit: what it measures, how often together, and how evenly.

    A plain record written to ``units.csv``; nothing reads it from outside the process.

    Attributes:
        corpus: The corpus the unit belongs to.
        group: The part of the corpus it belongs to (``group_of``).
        unit: Its key.
        span: Time between its first and last observation, in the corpus's time unit.
        channels: Channels it measures at all.
        channel_share: Those channels' share of the channels its part measures.
        observations: Its observations.
        instants: The distinct instants it is observed at.
        co_observation: The mean, over its instants, of the share of its channels read at that
            instant: one on a grid, one over the number of channels where no two channels share
            an instant.
        gap_variation: The median over channels of the coefficient of variation of the spacing
            between a channel's readings; zero on a grid.
        rate_spread: The decimal logarithm of the ratio between the slowest and the fastest
            channel's median spacing; zero where every channel runs at one rate.
    """

    corpus: str
    group: str
    unit: str
    span: float
    channels: int
    channel_share: float
    observations: int
    instants: int
    co_observation: float
    gap_variation: float
    rate_spread: float


@dataclass(frozen=True)
class WindowRow:
    """What one window holds; a plain record written to ``windows.csv``.

    Attributes:
        corpus: The corpus the window was cut from.
        unit: The unit it was cut from.
        start: Where it starts, in the corpus's time unit.
        tokens: Observations inside it.
        channel_share: Share of its unit's part's channels it holds at least once.
        per_channel: Median observations per channel it holds.
        once: Share of the channels it holds that it holds exactly once.
    """

    corpus: str
    unit: str
    start: float
    tokens: int
    channel_share: float
    per_channel: float
    once: float


def chosen_units(reader: CorpusReader, units: int, seed: int) -> list[CorpusUnit]:
    """At most ``units`` of the corpus's units, by seed, in the reader's own order.

    The reader's order is kept so that a reader holding one file of units at a time (the
    satellite missions) reads each file once.
    """
    every = list(reader.read_units())
    kept = chosen_keys((str(unit.key) for unit in every), units, seed)
    return [unit for unit in every if str(unit.key) in kept]


def chosen_keys(keys: Iterable[str], units: int, seed: int) -> set[str]:
    """At most ``units`` of the keys, by seed; the same keys draw the same units anywhere."""
    return set(sorted(keys, key=lambda key: seeded_rank(seed, key))[:units])


def group_of(key: str) -> str:
    """The part of the corpus a unit belongs to: a subset, a mission, a set of stays.

    The parts of one corpus may measure different channels (a mission its own, a C-MAPSS subset
    read per condition six times as many), so a unit's share of channels is counted within its
    part.
    """
    return key.split("/", 1)[0] if "/" in key else ""


def group_widths(read: Iterable[tuple[str, Times]]) -> dict[str, int]:
    """The channels each part's units measure at all, by part."""
    channels: dict[str, set[str]] = defaultdict(set)
    for key, times in read:
        channels[group_of(key)].update(times)
    return {group: len(held) for group, held in channels.items()}


def unit_rows_of(corpus: str, read: Sequence[tuple[str, Times]]) -> list[UnitRow]:
    """A row per unit, its share of channels counted against its part's units in ``read``."""
    widths = group_widths(read)
    return [unit_row(corpus, key, times, widths[group_of(key)]) for key, times in read]


def unit_row(corpus: str, key: str, times: Times, group_channels: int) -> UnitRow:
    observed = {channel: t for channel, t in times.items() if t.size}
    instants: Counter[float] = Counter()
    variation: list[float] = []
    spacing: list[float] = []
    for t in observed.values():
        distinct = np.unique(t)
        instants.update(distinct.tolist())
        gaps = np.diff(distinct)
        if gaps.size:
            spacing.append(float(np.median(gaps)))
        if gaps.size >= 2:
            variation.append(float(gaps.std() / gaps.mean()))
    every = np.concatenate(list(observed.values())) if observed else np.empty(0)
    count = len(observed)
    return UnitRow(
        corpus=corpus,
        group=group_of(key),
        unit=key,
        span=float(every.max() - every.min()) if every.size else 0.0,
        channels=count,
        channel_share=count / group_channels,
        observations=int(every.size),
        instants=len(instants),
        co_observation=(
            statistics.fmean(n / count for n in instants.values()) if instants else math.nan
        ),
        gap_variation=statistics.median(variation) if variation else math.nan,
        rate_spread=(math.log10(max(spacing) / min(spacing)) if len(spacing) >= 2 else math.nan),
    )


def window_rows(
    corpus: str,
    unit: CorpusUnit,
    times: Times,
    window: WindowSpec,
    group_channels: int,
    per_unit: int,
    seed: int,
) -> list[WindowRow]:
    """At most ``per_unit`` of the unit's windows, by seed; a window holding nothing is none."""
    rows = []
    for extent in window.windows_over(unit.extent):
        inside = {
            channel: int(((t >= extent.start) & (t < extent.end)).sum())
            for channel, t in times.items()
        }
        held = [n for n in inside.values() if n]
        if not held:
            continue
        rows.append(
            WindowRow(
                corpus=corpus,
                unit=str(unit.key),
                start=extent.start,
                tokens=sum(held),
                channel_share=len(held) / group_channels,
                per_channel=float(statistics.median(held)),
                once=sum(1 for n in held if n == 1) / len(held),
            )
        )
    rows.sort(key=lambda row: seeded_rank(seed, row.unit, row.start))
    return rows[:per_unit]


def times_by_channel(reader: CorpusReader, unit: CorpusUnit) -> dict[str, NDArray[np.float64]]:
    collected: dict[str, list[float]] = defaultdict(list)
    for observation in reader.read_observations(unit.key):
        collected[observation.channel].append(observation.time)
    return {channel: np.asarray(t, dtype=np.float64) for channel, t in collected.items()}


def root_of(source: Source, sample: bool) -> tuple[Path, bool]:
    """Where the corpus is read from, and whether that is the miniature sample."""
    root = None if sample else raw_root(source.raw)
    if root is not None:
        return root, False
    miniature = SAMPLES / source.raw
    return (miniature / "UTSD-12G" if source.raw == "utsd" else miniature), True


def measure_source(
    source: Source, units: int, windows: int, seed: int, sample: bool
) -> tuple[list[UnitRow], list[WindowRow], int, bool]:
    """The units and windows of one corpus, and how many of the units drawn hold nothing.

    A unit holding no observation yields no window, so the publication lists it as empty and it
    is left out here too. A channel's share is counted against the channels the sampled units of
    the same part measure at all, not the reader's schema: describing a corpus hashes every byte
    of it, and the satellite corpus alone holds twenty gigabytes.
    """
    root, from_sample = root_of(source, sample)
    reader = source.reader(root)
    drawn = [(unit, times_by_channel(reader, unit)) for unit in chosen_units(reader, units, seed)]
    read = [(unit, times) for unit, times in drawn if times]
    keyed = [(str(unit.key), times) for unit, times in read]
    widths = group_widths(keyed)
    rows: list[WindowRow] = []
    for unit, times in read:
        width = widths[group_of(str(unit.key))]
        rows.extend(window_rows(source.corpus, unit, times, source.window, width, windows, seed))
    return unit_rows_of(source.corpus, keyed), rows, len(drawn) - len(read), from_sample


def quantiles(values: Iterable[float]) -> tuple[float, float, float]:
    """The 10th, 50th and 90th percentiles of the finite values; not a number where none is."""
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return math.nan, math.nan, math.nan
    low, middle, high = np.percentile(finite, [10, 50, 90])
    return float(low), float(middle), float(high)


UNIT_MEASURES = ("channel_share", "co_observation", "gap_variation", "rate_spread")
WINDOW_MEASURES = ("tokens", "channel_share", "per_channel", "once")


def corpus_rows(
    unit_rows: Sequence[UnitRow], rows: Sequence[WindowRow]
) -> list[dict[str, str | int | float]]:
    """One row per corpus: its regime and the 10th, 50th and 90th percentile of every measure."""
    result = []
    for corpus in dict.fromkeys(row.corpus for row in unit_rows):
        units = [row for row in unit_rows if row.corpus == corpus]
        windows = [row for row in rows if row.corpus == corpus]
        summary: dict[str, str | int | float] = {
            "corpus": corpus,
            "regime": regime_of(corpus).value,
            "units": len(units),
            "windows": len(windows),
        }
        levels: tuple[tuple[str, Sequence[UnitRow | WindowRow], tuple[str, ...]], ...] = (
            ("unit", units, UNIT_MEASURES),
            ("window", windows, WINDOW_MEASURES),
        )
        for level, held, measures in levels:
            for name in measures:
                low, middle, high = quantiles(float(getattr(row, name)) for row in held)
                summary |= {
                    f"{level}_{name}_p10": low,
                    f"{level}_{name}_p50": middle,
                    f"{level}_{name}_p90": high,
                }
        result.append(summary)
    return result


def measure(
    out: Path,
    sources: Sequence[Source] = SOURCES,
    units: int = 100,
    windows: int = 20,
    seed: int = 1,
    sample: bool = False,
) -> list[str]:
    """Measure every source into ``out``; returns a line per source saying what was read."""
    out.mkdir(parents=True, exist_ok=True)
    unit_rows: list[UnitRow] = []
    rows: list[WindowRow] = []
    said = []
    for source in sources:
        held_units, held_windows, empty, from_sample = measure_source(
            source, units, windows, seed, sample
        )
        unit_rows.extend(held_units)
        rows.extend(held_windows)
        read = "miniature sample" if from_sample else "raw corpus"
        said.append(
            f"{source.corpus}: {len(held_units)} units, {empty} empty, "
            f"{len(held_windows)} windows ({read})"
        )
        print(said[-1], flush=True)
    write(out / "units.csv", UNIT_COLUMNS, [asdict(row) for row in unit_rows])
    write(out / "windows.csv", WINDOW_COLUMNS, [asdict(row) for row in rows])
    summary = corpus_rows(unit_rows, rows)
    write(out / "corpora.csv", tuple(summary[0]), summary)
    for line in corpus_table(summary):
        print(line)
    return said


TABLE_MEASURES = (
    ("unit_co_observation", "Co-observed", ".2f"),
    ("unit_gap_variation", "Gap variation", ".2f"),
    ("unit_rate_spread", "Rate spread", ".2f"),
    ("unit_channel_share", "Channels / unit", ".2f"),
    ("window_tokens", "Tokens / window", ",.0f"),
    ("window_per_channel", "Per channel", ".0f"),
    ("window_once", "Read once", ".2f"),
)


def corpus_table(summary: Sequence[Mapping[str, str | int | float]]) -> list[str]:
    """The median of every measure per corpus, its 10th and 90th percentiles in brackets."""
    head = ["Corpus", "Regime", *(title for _, title, _ in TABLE_MEASURES)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in summary:
        cells = [str(row["corpus"]), str(row["regime"])]
        for key, _, spec in TABLE_MEASURES:
            low, middle, high = (float(row[f"{key}_{q}"]) for q in ("p10", "p50", "p90"))
            cells.append(
                "—" if math.isnan(middle) else f"{middle:{spec}} [{low:{spec}}; {high:{spec}}]"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return lines


# --- exposure ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Published:
    """What a published corpus holds that its share of a pretraining depends on.

    A plain record read off a manifest; nothing reads it from outside the process.

    Attributes:
        windows: Windows the block holds, both sides.
        training_share: Share of the corpus's windowed units on the training side.
        observed: Observed values of the training side, as the channel statistics count them.
    """

    windows: int
    training_share: float
    observed: int


@dataclass(frozen=True)
class Experiment:
    """The corpora a pretraining reads, and how often and how much of each.

    A plain record read off an experiment file; nothing reads it from outside the process.
    """

    name: str
    corpora: tuple[str, ...]
    passes: Mapping[str, int]
    fractions: Mapping[str, float]

    @classmethod
    def read(cls, path: Path) -> Self:
        with path.open("rb") as file:
            document = tomllib.load(file)
        return cls(
            document["name"],
            tuple(document["corpora"]),
            dict(document.get("passes", {})),
            dict(document.get("fraction", {})),
        )


@dataclass(frozen=True)
class ExposureRow:
    """One corpus's share of a pretraining, or one regime's (``corpus`` ``*``).

    A plain record written to ``exposure.csv``; nothing reads it from outside the process.
    """

    experiment: str
    corpus: str
    regime: str
    passes: int | None
    fraction: float | None
    windows: float
    observed: float
    window_share: float
    observed_share: float


def exposure_rows(experiment: Experiment, published: Mapping[str, Published]) -> list[ExposureRow]:
    """Each corpus's share of the pretraining's windows and observed values, then per regime.

    A run reads the training windows of a corpus once per pass, of the fraction of its training
    units it is given; a step is taken over windows of one corpus, so a corpus's share of the
    windows is its share of the steps. The fraction is applied to windows and values as to units,
    which holds on average over the units a seed draws.
    """
    read = []
    for corpus in experiment.corpora:
        facts = published[corpus]
        passes = experiment.passes.get(corpus, 1)
        fraction = experiment.fractions.get(corpus, 1.0)
        windows = facts.windows * facts.training_share * fraction * passes
        read.append((corpus, passes, fraction, windows, facts.observed * fraction * passes))
    all_windows = sum(row[3] for row in read)
    all_observed = sum(row[4] for row in read)
    rows = [
        ExposureRow(
            experiment=experiment.name,
            corpus=corpus,
            regime=regime_of(corpus).value,
            passes=passes,
            fraction=fraction,
            windows=windows,
            observed=observed,
            window_share=windows / all_windows,
            observed_share=observed / all_observed,
        )
        for corpus, passes, fraction, windows, observed in read
    ]
    for regime in SamplingRegime:
        held = [row for row in rows if row.regime == regime.value]
        rows.append(
            ExposureRow(
                experiment=experiment.name,
                corpus="*",
                regime=regime.value,
                passes=None,
                fraction=None,
                windows=sum(row.windows for row in held),
                observed=sum(row.observed for row in held),
                window_share=sum(row.window_share for row in held),
                observed_share=sum(row.observed_share for row in held),
            )
        )
    return rows


def manifests(path: Path) -> Iterator[tuple[str, str, str]]:
    """The lines ``name key checksum`` of a file, comments and blank lines skipped."""
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, key, checksum = line.split()[:3]
            yield name, key, checksum


def read_published(path: Path) -> dict[str, Published]:  # pragma: no cover - reads the bucket
    from emblema.catalog.contracts.published_corpus_manifest_json import (
        PublishedCorpusManifestJson,
    )
    from emblema.config.settings import Settings
    from emblema.entrypoints.configured import configured_store
    from emblema.shared.kernel.artifacts import ArtifactRef
    from emblema.shared.kernel.checksums import Checksum

    store = configured_store(Settings())
    codec = PublishedCorpusManifestJson()
    result = {}
    for name, key, checksum in manifests(path):
        manifest = codec.decode(store.get(ArtifactRef(key=key, checksum=Checksum.parse(checksum))))
        windowed = set(manifest.units)
        result[name] = Published(
            windows=manifest.window_count,
            training_share=len(windowed & set(manifest.training_units)) / len(windowed),
            observed=sum(c.statistics.count for c in manifest.channels if c.statistics),
        )
    return result


def exposure(out: Path, published: Mapping[str, Published], experiments: Sequence[Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = [row for path in experiments for row in exposure_rows(Experiment.read(path), published)]
    write(out / "exposure.csv", EXPOSURE_COLUMNS, [asdict(row) for row in rows])
    print("| Pretraining | Regime | Share of windows | Share of observed values |")
    print("|---|---|---:|---:|")
    for row in rows:
        if row.corpus == "*":
            print(
                f"| {row.experiment} | {row.regime} | {row.window_share:.1%} | "
                f"{row.observed_share:.1%} |"
            )


# --- common --------------------------------------------------------------------------------------

UNIT_COLUMNS = tuple(field.name for field in fields(UnitRow))
WINDOW_COLUMNS = tuple(field.name for field in fields(WindowRow))
EXPOSURE_COLUMNS = tuple(field.name for field in fields(ExposureRow))


def write(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:  # pragma: no cover - the command line
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    measuring = commands.add_parser("measure", help="the sampling of every corpus")
    measuring.add_argument("--out", type=Path, default=REPO_ROOT / "data/report/regime-coverage")
    measuring.add_argument("--units", type=int, default=100, help="units read per corpus")
    measuring.add_argument("--windows", type=int, default=20, help="windows kept per unit")
    measuring.add_argument("--seed", type=int, default=1)
    measuring.add_argument("--corpus", nargs="*", help="only these corpora, the portal included")
    exposing = commands.add_parser("exposure", help="each regime's share of each pretraining")
    exposing.add_argument("--out", type=Path, default=REPO_ROOT / "data/report/regime-coverage")
    exposing.add_argument("--manifests", type=Path, required=True)
    exposing.add_argument("--experiments", type=Path, nargs="+", required=True)
    arguments = parser.parse_args()
    if arguments.command == "measure":
        named = (*SOURCES, *TASK_REGIME_SOURCES)
        chosen = [s for s in named if s.corpus in arguments.corpus] if arguments.corpus else SOURCES
        measure(arguments.out, chosen, arguments.units, arguments.windows, arguments.seed)
    else:
        exposure(arguments.out, read_published(arguments.manifests), arguments.experiments)


if __name__ == "__main__":
    main()
