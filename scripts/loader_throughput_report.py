"""Measure whether feeding the model costs less than running it, and print the numbers as a note.

The loader's tests assert what must hold everywhere — the order a seed gives, the shapes a batch
has. This script measures what only holds on a machine: how fast windows become batches, what the
windows cost to keep in memory, and how long a training step takes beside them. Run it once per
machine, because that is what the numbers depend on:

    uv sync --all-extras
    uv run scripts/loader_throughput_report.py

The output is markdown, meant to be pasted under a dated heading in the verification note. The
model is the encoder at the shape of each compute tier in the budget file: what a step costs is set
by width, depth and token count, not by what the weights have learned.
"""

import io
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from itertools import chain, count, islice
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from emblema.catalog.adapters.readers.cmapss import CmapssCorpusReader
from emblema.catalog.adapters.tokenisation.sliding_window import SlidingWindowTokeniser
from emblema.catalog.domain.channels.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.tokenisation.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.config.compute_tiers import ComputeTierProfile, ComputeTiers
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.encoder.tier_architecture import architecture_of
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.shared.adapters.loaders.seeded_shuffle_sampler import SeededShuffleSampler
from emblema.shared.adapters.loaders.window_loader import WindowLoader
from emblema.shared.kernel.tokens import Token, TokenWindow
from scripts.budget_file import budget
from scripts.reporting import table

RAW = REPO_ROOT / "data" / "raw" / "cmapss"
CORPUS = "cmapss"

# Enough windows to time batching without holding a corpus of Python objects in memory; the cost of
# holding one is itself one of the numbers below.
WINDOWS = 512
BATCH_SIZE = 8
EPOCHS = 3
STEPS = 8
# Corpus sizes the epoch order is timed over: the sample, the full C-MAPSS window count measured by
# the data spike, and a corpus an order of magnitude larger.
ORDERED_SIZES = (WINDOWS, 25_395, 250_000)
# Worker counts the collation cost is reported for. Zero is the configuration a training run uses
# until the corpus is a memory map; the others say what handing windows to a process costs today.
WORKER_COUNTS = (0, 2)
# How much cheaper feeding the model must be than running it before the loader is off the critical
# path. A floor, not a target: it measured between 30x and 65x when this was written, and a fall to
# single digits is the signal to move the work off the training process.
MARGIN = 2.0


def default_window() -> WindowSpec:
    windows = budget()["corpora"][CORPUS]["windows"]
    spec = next(window for window in windows if window.get("default"))
    return WindowSpec(spec["length"], spec["stride"])


def channel_count() -> int:
    """Channels the corpus carries: as measured if the data spike has run, as estimated if not."""
    corpus = budget()["corpora"][CORPUS]
    return int(corpus.get("measured", corpus["estimate"])["channels"])


def raw_root() -> Path | None:
    hits = sorted(RAW.rglob("train_FD001.txt")) if RAW.is_dir() else []
    return hits[0].parent if hits else None


def real_windows(root: Path, limit: int) -> Iterator[TokenWindow]:
    """The first ``limit`` windows of FD001, through the reader and the tokeniser."""
    reader = CmapssCorpusReader(root, ("FD001",))
    tokeniser = SlidingWindowTokeniser()
    units = list(reader.read_units())
    scheme = TokenisationScheme.for_vocabulary(ChannelVocabulary()).extended_with(
        CORPUS, reader.describe().channel_schema
    )
    scheme = tokeniser.fit(
        CORPUS,
        chain.from_iterable(reader.read_observations(unit.key) for unit in units),
        (),
        scheme,
    )
    window = default_window()
    produced = chain.from_iterable(
        tokeniser.tokenise(CORPUS, unit, reader.read_observations(unit.key), scheme, window)
        for unit in units
    )
    return islice((placed.window for placed in produced), limit)


def synthetic_windows(total: int, window: WindowSpec) -> Iterator[TokenWindow]:
    """Windows of the shape the default C-MAPSS window has, for a machine without the raw data.

    C-MAPSS reports every channel once per cycle, so this window's length in cycles is also its
    count of samples. That equality holds for this corpus and nowhere else: a window spec measures
    elapsed time, and nothing outside this stand-in may read a sample count out of one.
    """
    channels = channel_count()
    samples = int(window.length)
    for index in range(total):
        yield TokenWindow.of(
            Token(
                channel_id=1 + channel,
                value=float((index + step + channel) % 97) / 97.0,
                time=step / samples,
                # The gap reaches back to the previous token of the same channel, which the
                # first token of a window has not got: it carries no gap rather than one step.
                gap=(1 / samples) if step else 0.0,
            )
            for step in range(samples)
            for channel in range(channels)
        )


@dataclass(frozen=True)
class Corpus:
    """The windows a run is timed over, and what holding them cost."""

    source: str
    windows: list[TokenWindow]
    bytes_held: int

    @property
    def median_tokens(self) -> int:
        return int(statistics.median(len(window) for window in self.windows))


def load_corpus() -> Corpus:
    root = raw_root()
    produce = real_windows(root, WINDOWS) if root else synthetic_windows(WINDOWS, default_window())
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    windows = list(produce)
    held = tracemalloc.get_traced_memory()[0] - before
    tracemalloc.stop()
    source = "C-MAPSS FD001, default window" if root else "synthetic, shape of the default window"
    return Corpus(source, windows, held)


def synchronise(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def median_seconds(step: Callable[[], object], repeats: int) -> float:
    """The median of ``repeats`` timings, the first one thrown away as warm-up."""
    timings: list[float] = []
    for _ in range(repeats + 1):
        started = time.perf_counter()
        step()
        timings.append(time.perf_counter() - started)
    return statistics.median(timings[1:])


def collate_seconds(windows: Sequence[TokenWindow], *, num_workers: int) -> float:
    """Seconds per batch, measured over whole epochs.

    Worker processes prepare batches ahead of the one being asked for, so timing a single ``next``
    would measure how full a queue is rather than what the work costs; an epoch also carries the
    price of starting those workers, which is paid once per epoch and belongs in the number.
    """
    loader = WindowLoader(
        windows, batch_size=BATCH_SIZE, seed=1, num_workers=num_workers, drop_last=True
    )

    def epoch() -> None:
        for _ in loader.batches_of(0):
            pass

    return median_seconds(epoch, EPOCHS) / len(loader)


def order_seconds(size: int) -> float:
    """Seconds to lay out one epoch's order, timed on a fresh epoch each repeat."""
    sampler = SeededShuffleSampler(size, seed=1)
    epochs = count(1)

    def order() -> None:
        sampler.set_epoch(next(epochs))
        list(sampler)

    return median_seconds(order, 3)


def transfer_seconds(windows: Sequence[TokenWindow], *, device: str) -> float:
    """Seconds to move one collated batch onto ``device``.

    Feeding the model is collating plus this: a batch built in host memory has to cross to the
    accelerator before the step can start, and on a unified-memory device that crossing is cheap
    but not free. Leaving it out of the comparison would flatter the loader.
    """
    loader = WindowLoader(windows, batch_size=BATCH_SIZE, seed=1, drop_last=True)
    batch = next(loader.batches_of(0))

    def transfer() -> None:
        batch.to(device)
        synchronise(device)

    return median_seconds(transfer, STEPS)


def step_seconds(
    windows: Sequence[TokenWindow], *, architecture: EncoderArchitecture, device: str
) -> float:
    """Seconds for one forward and backward pass over a batch already on ``device``."""
    model = SetEncoder.for_vocabulary(architecture, channel_count()).to(device)
    loader = WindowLoader(windows, batch_size=BATCH_SIZE, seed=1, drop_last=True)
    batch = next(loader.batches_of(0)).to(device)

    def step() -> None:
        model(*batch.args).pow(2).mean().backward()
        model.zero_grad(set_to_none=True)
        synchronise(device)

    return median_seconds(step, STEPS)


def device_in_use() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Measurements:
    """Everything one run of this script measured, before any of it is formatted."""

    device: str
    corpus: Corpus
    collating: dict[int, float]
    transfer: float
    steps: dict[str, float]
    tiers: list[ComputeTierProfile]

    def feeding(self, workers: int) -> float:
        """Seconds to put one batch in front of the model: collating it and moving it across."""
        return self.collating[workers] + self.transfer


def measure() -> Measurements:
    device = device_in_use()
    corpus = load_corpus()
    tiers = [tier for tier in ComputeTiers.load().tiers if tier.name in ("S", "M")]
    return Measurements(
        device=device,
        corpus=corpus,
        collating={
            workers: collate_seconds(corpus.windows, num_workers=workers)
            for workers in WORKER_COUNTS
        },
        transfer=transfer_seconds(corpus.windows, device=device),
        steps={
            tier.name: step_seconds(
                corpus.windows, architecture=architecture_of(tier), device=device
            )
            for tier in tiers
        },
        tiers=tiers,
    )


def heading(measured: Measurements) -> str:
    rows = [
        ("Machine", f"{platform.platform()}, {platform.processor() or 'unknown CPU'}"),
        ("Python", platform.python_version()),
        ("torch", version("torch")),
        ("numpy", version("numpy")),
        ("Device", measured.device),
        ("Batch", str(BATCH_SIZE)),
    ]
    return f"## {datetime.now():%Y-%m-%d} — {platform.system()} {platform.machine()}\n\n" + table(
        ("", ""), rows
    )


def windows_section(measured: Measurements) -> str:
    corpus = measured.corpus
    row = (
        corpus.source,
        str(len(corpus.windows)),
        str(corpus.median_tokens),
        f"{corpus.bytes_held / len(corpus.windows) / 1024:.1f} KiB",
    )
    return "### Windows\n\n" + table(
        ("Source", "Windows", "Tokens per window", "Held in memory per window"), [row]
    )


def feeding_section(measured: Measurements) -> str:
    tokens = measured.corpus.median_tokens
    rows = [
        (
            str(workers),
            f"{measured.collating[workers] * 1e3:.2f} ms",
            f"{measured.transfer * 1e3:.2f} ms",
            f"{BATCH_SIZE / measured.feeding(workers):,.0f}",
            f"{BATCH_SIZE * tokens / measured.feeding(workers):,.0f}",
        )
        for workers in WORKER_COUNTS
    ]
    return "### Feeding the model\n\n" + table(
        (
            "Workers",
            "Collating, s / batch",
            f"To {measured.device}, s / batch",
            "Windows / s",
            "Tokens / s",
        ),
        rows,
    )


def ordering_section() -> str:
    rows = [(f"{size:,}", f"{order_seconds(size) * 1e3:.1f} ms") for size in ORDERED_SIZES]
    return "### Ordering an epoch\n\n" + table(("Windows in the corpus", "s / epoch"), rows)


def steps_section(measured: Measurements) -> str:
    rows = [
        (
            tier.name,
            str(tier.width),
            str(tier.layers),
            f"{measured.steps[tier.name] * 1e3:.0f} ms",
            f"{BATCH_SIZE / measured.steps[tier.name]:,.0f}",
        )
        for tier in measured.tiers
    ]
    return "### Training step\n\n" + table(
        ("Tier", "Width", "Layers", "s / step", "Windows / s"), rows
    )


def verdict_section(measured: Measurements) -> str:
    """The answer the script exists for, stated per configuration rather than left to be divided."""
    lines = ["### Verdict", ""]
    for name, seconds in measured.steps.items():
        for workers in WORKER_COUNTS:
            ratio = seconds / measured.feeding(workers)
            outcome = "off the critical path" if ratio > MARGIN else "THE BOTTLENECK"
            lines.append(
                f"- Tier {name}, {workers} worker(s): a step costs {ratio:.1f}x feeding it "
                f"— {outcome} at a margin of {MARGIN:.0f}x."
            )
    return "\n".join(lines)


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        # The headings use an em dash, which a Windows console's default code page lacks, and
        # the note this output is pasted into is LF-only.
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    measured = measure()
    sections = (
        heading(measured),
        windows_section(measured),
        feeding_section(measured),
        ordering_section(),
        steps_section(measured),
        verdict_section(measured),
    )
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
