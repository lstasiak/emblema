"""Draw a raw window beside the reconstruction of its tokens, and say what the corpus looks like.

Pretraining rests on a token window still holding the measurements it was cut from. The report cuts
a few windows, reads them back through the tokenisation scheme and draws both curves per channel; it
prints the largest round-trip residual and, per channel, the fitted spread and how far and how often
values lie outside it.

    uv sync --all-extras
    uv run scripts/window_sanity_report.py --corpus cmapss --subset FD001

Prints markdown for the verification note and writes the figures beside it. Without the raw corpus
it uses the miniature test sample and says so. Statistics are fitted on every unit read: this checks
the data, and none of its numbers belongs in a result.
"""

import argparse
import io
import math
import platform
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from itertools import chain
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

# A report writes files and never opens a window; the backend has to be chosen before pyplot is.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emblema.catalog.adapters.readers.cmapss import SUBSETS as CMAPSS_SUBSETS
from emblema.catalog.adapters.readers.cmapss import CmapssCorpusReader
from emblema.catalog.adapters.readers.esa_ad import EsaAdCorpusReader
from emblema.catalog.adapters.readers.skab import SUBSETS as SKAB_SUBSETS
from emblema.catalog.adapters.readers.skab import SkabCorpusReader
from emblema.catalog.adapters.readers.smd import SUBSETS as SMD_SUBSETS
from emblema.catalog.adapters.readers.smd import SmdCorpusReader
from emblema.catalog.adapters.synthetic.layouts import CONTROL_PROCESS, LAYOUTS
from emblema.catalog.adapters.synthetic.sensor_layout import SensorLayout
from emblema.catalog.adapters.synthetic.synthetic_corpus_reader import SyntheticCorpusReader
from emblema.catalog.adapters.tokenisation.sliding_window import SlidingWindowTokeniser
from emblema.catalog.domain.channels.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.catalog.domain.tokenisation.placed_window import PlacedWindow
from emblema.catalog.domain.tokenisation.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.tokenisation.window_reconstruction import WindowReconstruction
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.catalog.ports.corpus_reader import CorpusReader
from scripts.budget_file import budget
from scripts.raw_corpora import raw_root
from scripts.reporting import table

RAW = REPO_ROOT / "data" / "raw"
SAMPLES = REPO_ROOT / "tests" / "data"
FIGURES = REPO_ROOT / "docs" / "verification" / "figures"

# A value this many fitted deviations from its channel's mean is not wrong, but it is worth a look:
# a sensor stuck at a rail, a unit column read into a measurement, a decimal point lost.
OUTLYING = 8.0
# What the round trip must hold to. Normalising and putting the value back is exact arithmetic
# reached by inexact means, so the bound is the resolution of the double, not zero.
TOLERANCE = 1e-6


def default_window(corpus: str) -> WindowSpec:
    """The window the budget file gives this corpus.

    Only the corpora the pretraining budget is arithmetic about are in that file; a generated one
    is not, and says so rather than failing on a missing key.
    """
    facts = budget()["corpora"].get(corpus)
    if facts is None:
        raise SystemExit(f"{corpus} has no window in the budget file; give --length and --stride")
    spec = next(window for window in facts["windows"] if window.get("default"))
    return WindowSpec(spec["length"], spec["stride"])


def cmapss_reader(root: Path, subset: str | None) -> CorpusReader:
    return CmapssCorpusReader(root, (subset,) if subset else CMAPSS_SUBSETS)


def skab_reader(root: Path, subset: str | None) -> CorpusReader:
    return SkabCorpusReader(root, (subset,) if subset else SKAB_SUBSETS)


def smd_reader(root: Path, subset: str | None) -> CorpusReader:
    return SmdCorpusReader(root, (subset,) if subset else SMD_SUBSETS)


def esa_ad_reader(root: Path, subset: str | None) -> CorpusReader:
    return EsaAdCorpusReader(root, (subset,) if subset else EsaAdCorpusReader.SUBSETS)


def generated_reader(layout: SensorLayout) -> Callable[[Path, str | None], CorpusReader]:
    """A reader of a generated corpus, which takes neither a directory nor a subset."""

    def read(_root: Path, _subset: str | None) -> CorpusReader:
        return SyntheticCorpusReader(CONTROL_PROCESS, layout)

    return read


# One entry per corpus that has an adapter; a new reader is a line here, not a change below.
READERS: dict[str, Callable[[Path, str | None], CorpusReader]] = {
    "cmapss": cmapss_reader,
    "skab": skab_reader,
    "smd": smd_reader,
    "esa_ad": esa_ad_reader,
    **{name: generated_reader(layout) for name, layout in LAYOUTS.items()},
}


def corpus_root(corpus: str) -> tuple[Path, str]:
    """Where the corpus is on this machine, and whether that is the real thing or the sample."""
    root = raw_root(corpus)
    if root is not None:
        return root, "raw corpus"
    return SAMPLES / corpus, "miniature sample"


def corpus_source(corpus: str) -> tuple[Path | None, str]:
    """Where the corpus is on this machine; a generated one is nowhere and needs no files."""
    if corpus in LAYOUTS:
        return None, "generated from its specification"
    return corpus_root(corpus)


# Plain records: nothing here is validated, serialised or read from outside the process, so they
# carry data between the measuring half of the script and the printing half and nothing more.
@dataclass(frozen=True)
class ChannelDiagnostic:
    """What the corpus says about one channel, beside what the scheme fitted for it."""

    channel: str
    count: int
    mean: float
    std: float
    lowest: float
    highest: float
    outlying: int


@dataclass(frozen=True)
class WindowCheck:
    """One window drawn and read back: where it came from and how far the round trip strayed."""

    unit: str
    extent: TimeExtent
    tokens: int
    value_residual: float
    time_residual: float
    figure: Path


@dataclass(frozen=True)
class Report:
    corpus: str
    source: str
    root: Path | None
    window: WindowSpec
    units: int
    short_units: int
    channels: list[ChannelDiagnostic]
    checks: list[WindowCheck]


def observations_of(reader: CorpusReader, units: Iterable[CorpusUnit]) -> Iterator[Observation]:
    return chain.from_iterable(reader.read_observations(unit.key) for unit in units)


def fitted_scheme(
    reader: CorpusReader, corpus: str, units: Sequence[CorpusUnit]
) -> TokenisationScheme:
    unfitted = TokenisationScheme.for_vocabulary(ChannelVocabulary()).extended_with(
        corpus, reader.describe().channel_schema
    )
    statics = chain.from_iterable(unit.static_features for unit in units)
    return SlidingWindowTokeniser().fit(corpus, observations_of(reader, units), statics, unfitted)


def diagnose_channels(
    reader: CorpusReader, corpus: str, units: Sequence[CorpusUnit], scheme: TokenisationScheme
) -> list[ChannelDiagnostic]:
    """The extremes and the far-out values of every channel, in the deviations the scheme fitted."""
    vocabulary = scheme.vocabulary
    extremes: dict[str, list[float]] = {}
    outlying: dict[str, int] = {}
    timed = ((o.channel, o.value) for o in observations_of(reader, units))
    timeless = ((f.channel, f.value) for unit in units for f in unit.static_features)
    for channel, value in chain(timed, timeless):
        normalised = scheme.statistics_of(vocabulary.id_of(corpus, channel)).normalise(value)
        seen = extremes.setdefault(channel, [normalised, normalised])
        seen[0] = min(seen[0], normalised)
        seen[1] = max(seen[1], normalised)
        if abs(normalised) > OUTLYING:
            outlying[channel] = outlying.get(channel, 0) + 1
    diagnostics = []
    for entry in vocabulary.entries_of(corpus):
        statistics = scheme.statistics_of(entry.channel_id)
        lowest, highest = extremes.get(entry.channel, [0.0, 0.0])
        diagnostics.append(
            ChannelDiagnostic(
                entry.channel,
                statistics.count,
                statistics.mean,
                statistics.std,
                lowest,
                highest,
                outlying.get(entry.channel, 0),
            )
        )
    return diagnostics


def spread_over(placed: Sequence[PlacedWindow], count: int) -> list[PlacedWindow]:
    """``count`` windows spaced along the unit, so the drawing is not all of one end of its life."""
    if count >= len(placed):
        return list(placed)
    step = (len(placed) - 1) / (count - 1) if count > 1 else 0.0
    return [placed[round(index * step)] for index in range(count)]


def refuse_renamed_channels(
    expected: Sequence[tuple[str, *tuple[float, ...]]],
    actual: Sequence[tuple[str, *tuple[float, ...]]],
    noun: str,
) -> None:
    """Stop the report where the round trip gave a value back under another channel's name.

    The two sides below are paired by their place in the sorted order, which only pairs a value
    with the value it came from while both sides carry the same channels. A channel returned as
    another one of equal value would otherwise be read as no error at all.
    """
    was = Counter(channel for channel, *_ in expected)
    now = Counter(channel for channel, *_ in actual)
    if was != now:
        strayed = sorted((was - now) + (now - was))
        raise SystemExit(f"the window read back {noun} of other channels: {', '.join(strayed)}")


def residual(
    inside: Sequence[Observation],
    statics: Sequence[StaticFeature],
    reconstruction: WindowReconstruction,
) -> tuple[float, float]:
    """How far the round trip strayed, in the corpus's own units of value and of time.

    The static features of the unit go through the window as tokens of their own and come back
    with it, so they are held to the same tolerance as the measurements; only they have no time
    to stray in.
    """
    expected = sorted((o.channel, o.time, o.value) for o in inside)
    actual = sorted((o.channel, o.time, o.value) for o in reconstruction.observations)
    if len(expected) != len(actual):
        raise SystemExit(
            f"the window read back {len(actual)} observations where {len(expected)} went in"
        )
    refuse_renamed_channels(expected, actual, "observations")
    was_static = sorted((f.channel, f.value) for f in statics)
    now_static = sorted((f.channel, f.value) for f in reconstruction.static_features)
    if len(was_static) != len(now_static):
        raise SystemExit(
            f"the window read back {len(now_static)} static features where {len(was_static)} "
            f"went in"
        )
    refuse_renamed_channels(was_static, now_static, "static features")
    times = [abs(was[1] - now[1]) for was, now in zip(expected, actual, strict=True)]
    values = [abs(was[2] - now[2]) for was, now in zip(expected, actual, strict=True)]
    values += [abs(was[1] - now[1]) for was, now in zip(was_static, now_static, strict=True)]
    return max(values, default=0.0), max(times, default=0.0)


def draw(
    corpus: str,
    unit: CorpusUnit,
    inside: Sequence[Observation],
    reconstruction: WindowReconstruction,
    extent: TimeExtent,
    destination: Path,
) -> Path:
    """One panel per channel: what the reader delivered, and what the tokens gave back."""
    channels = sorted({observation.channel for observation in inside})
    columns = min(5, len(channels))
    rows = math.ceil(len(channels) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.2 * columns, 2.2 * rows), squeeze=False, layout="constrained"
    )
    for index, channel in enumerate(channels):
        axis = axes[index // columns][index % columns]
        raw = [(o.time, o.value) for o in inside if o.channel == channel]
        back = [(o.time, o.value) for o in reconstruction.observations if o.channel == channel]
        axis.plot(*zip(*raw, strict=True), color="#1f77b4", linewidth=1.0, label="raw")
        axis.plot(
            *zip(*back, strict=True),
            linestyle="none",
            marker="o",
            markersize=3.5,
            markerfacecolor="none",
            color="#d62728",
            label="from tokens",
        )
        axis.set_title(channel, fontsize=9)
        axis.tick_params(labelsize=7)
        # Without this a narrow range is drawn as an offset printed over the channel's name.
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    for index in range(len(channels), rows * columns):
        axes[index // columns][index % columns].axis("off")
    axes[0][0].legend(fontsize=7, loc="best")
    statics = ", ".join(
        f"{feature.channel}={feature.value:g}" for feature in reconstruction.static_features
    )
    figure.suptitle(
        f"{corpus} · unit {unit.key} · window [{extent.start:g}, {extent.end:g}) · "
        f"{len(inside)} observations" + (f" · timeless: {statics}" if statics else ""),
        fontsize=10,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120)
    plt.close(figure)
    return destination


def chosen_window(arguments: argparse.Namespace, corpus: str) -> WindowSpec:
    if bool(arguments.length) != bool(arguments.stride):
        raise SystemExit("a window is a length and a stride; give both or neither")
    if arguments.length:
        return WindowSpec(arguments.length, arguments.stride)
    return default_window(corpus)


def chosen_unit(units: Sequence[CorpusUnit], key: str | None) -> CorpusUnit:
    """The unit asked for, or the longest one, where a time axis that drifts shows most."""
    if key is None:
        return max(units, key=lambda unit: unit.extent.length)
    for unit in units:
        if str(unit.key) == key:
            return unit
    known = ", ".join(str(unit.key) for unit in units[:5])
    raise SystemExit(f"no unit {key!r}; the corpus holds {known} and {len(units) - 5} more")


def measure(arguments: argparse.Namespace) -> Report:
    corpus = arguments.corpus
    root, source = corpus_source(corpus)
    # Only a generated corpus comes back without a directory, and the reader of one ignores what
    # it is handed; the placeholder keeps the table's readers alike rather than splitting it.
    reader = READERS[corpus](root or RAW / corpus, arguments.subset)
    units = list(reader.read_units())
    scheme = fitted_scheme(reader, corpus, units)
    window = chosen_window(arguments, corpus)
    chosen = chosen_unit(units, arguments.unit)
    observations = list(reader.read_observations(chosen.key))
    placed = list(SlidingWindowTokeniser().tokenise(corpus, chosen, observations, scheme, window))
    if not placed:
        raise SystemExit(f"unit {chosen.key} of {corpus} holds no window of {window}")
    checks = []
    for number, item in enumerate(spread_over(placed, arguments.windows), start=1):
        inside = [o for o in observations if item.extent.contains(o.time)]
        reconstruction = scheme.reconstruct(item.window, item.extent)
        values, times = residual(inside, chosen.static_features, reconstruction)
        figure = draw(
            corpus,
            chosen,
            inside,
            reconstruction,
            item.extent,
            Path(arguments.out) / f"{corpus}-{str(chosen.key).replace('/', '-')}-w{number}.png",
        )
        checks.append(WindowCheck(str(chosen.key), item.extent, len(inside), values, times, figure))
    return Report(
        corpus=corpus,
        source=source,
        root=root,
        window=window,
        units=len(units),
        short_units=sum(1 for unit in units if unit.extent.length < window.length),
        channels=diagnose_channels(reader, corpus, units, scheme),
        checks=checks,
    )


def shown(path: Path) -> str:
    """A path as the note carries it: relative to the repository when it lies inside one."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def where(root: Path | None) -> str:
    """The directory a corpus was read from, where there is one."""
    return "" if root is None else f", `{shown(root)}`"


def heading(report: Report) -> str:
    rows = [
        ("Machine", f"{platform.platform()}, {platform.processor() or 'unknown CPU'}"),
        ("Python", platform.python_version()),
        ("matplotlib", version("matplotlib")),
        ("Corpus", f"{report.corpus} ({report.source}{where(report.root)})"),
        ("Window", f"length {report.window.length:g}, stride {report.window.stride:g}"),
        ("Units", f"{report.units}, of which {report.short_units} shorter than a window"),
    ]
    return f"## {datetime.now():%Y-%m-%d} — {platform.system()} {platform.machine()}\n\n" + table(
        ("", ""), rows
    )


def round_trip_section(report: Report) -> str:
    rows = [
        (
            f"{check.unit}",
            f"[{check.extent.start:g}, {check.extent.end:g})",
            f"{check.tokens:,}",
            f"{check.value_residual:.2e}",
            f"{check.time_residual:.2e}",
            f"`{shown(check.figure)}`",
        )
        for check in report.checks
    ]
    return "### Windows drawn\n\n" + table(
        ("Unit", "Window", "Observations", "Largest value error", "Largest time error", "Figure"),
        rows,
    )


def channels_section(report: Report) -> str:
    rows = [
        (
            diagnostic.channel,
            f"{diagnostic.count:,}",
            f"{diagnostic.mean:.4g}",
            f"{diagnostic.std:.4g}" + (" (never varied)" if diagnostic.std == 0.0 else ""),
            f"{diagnostic.lowest:.2f}",
            f"{diagnostic.highest:.2f}",
            f"{diagnostic.outlying:,}",
        )
        for diagnostic in report.channels
    ]
    return "### Channels\n\n" + table(
        (
            "Channel",
            "Values",
            "Mean",
            "Spread",
            "Lowest, deviations",
            "Highest, deviations",
            f"Beyond {OUTLYING:g} deviations",
        ),
        rows,
    )


def verdict_section(report: Report) -> str:
    """The two questions the script exists to answer, answered rather than left to be divided."""
    worst = max(
        (max(check.value_residual, check.time_residual) for check in report.checks), default=0.0
    )
    held = "holds" if worst <= TOLERANCE else "DOES NOT HOLD"
    lines = [
        "### Verdict",
        "",
        f"- The round trip {held}: the largest error over {len(report.checks)} window(s) is "
        f"{worst:.2e}, against a tolerance of {TOLERANCE:.0e}.",
    ]
    constant = [c.channel for c in report.channels if c.std == 0.0]
    far_out = [c.channel for c in report.channels if c.outlying]
    lines.append(
        f"- Channels that never varied in the data fitted: {', '.join(constant) or 'none'}."
    )
    lines.append(
        f"- Channels holding values beyond {OUTLYING:g} deviations: {', '.join(far_out) or 'none'}."
    )
    lines.append("- Whether the two curves agree is a judgement made by looking at the figures.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", default="cmapss", choices=sorted(READERS))
    parser.add_argument(
        "--subset",
        help="one part of the corpus, where it has any: a C-MAPSS subset, a SKAB folder, "
        "an SMD machine group, an ESA mission",
    )
    parser.add_argument("--unit", help="unit to draw; the longest one by default")
    parser.add_argument("--windows", type=int, default=3, help="windows to draw, spread along it")
    parser.add_argument("--length", type=float, help="window length; the corpus default if unset")
    parser.add_argument("--stride", type=float, help="window stride; the corpus default if unset")
    parser.add_argument("--out", default=str(FIGURES), help="directory the figures are written to")
    arguments = parser.parse_args()
    if isinstance(sys.stdout, io.TextIOWrapper):
        # The headings use characters a Windows console's default code page lacks, and the note
        # this output is pasted into is LF-only.
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    report = measure(arguments)
    sections = (
        heading(report),
        round_trip_section(report),
        channels_section(report),
        verdict_section(report),
    )
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
