"""Measure whether feeding the model costs less than running it, and print the numbers as a note.

The loader's tests assert what must hold everywhere — the order a seed gives, the shapes a batch
has. This script measures what only holds on a machine: how fast windows become batches, what the
windows cost to keep in memory, and how long a training step takes beside them. Run it once per
machine, because that is what the numbers depend on:

    uv sync --all-extras
    uv run scripts/loader_throughput_report.py

The output is markdown, meant to be pasted under a dated heading in the verification note. The
model is the stand-in encoder from the export suite, sized from the compute tiers of the budget
file: the real encoder does not exist yet, and a step's cost is set by its width, depth and token
count rather than by the behaviour it learns.
"""

import io
import platform
import statistics
import sys
import time
import tomllib
import tracemalloc
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from itertools import chain, islice
from pathlib import Path
from typing import Any

# Run from anywhere: the stand-in encoder lives in the test package at the repository root, next to
# this directory. The imports below follow, which is why this file is exempt from the import-order
# rule in the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from emblema.catalog.adapters.readers.cmapss import CmapssCorpusReader
from emblema.catalog.adapters.tokenisation.sliding_window import SlidingWindowTokeniser
from emblema.catalog.domain.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.window_spec import WindowSpec
from emblema.pretraining.adapters.loaders.seeded_shuffle_sampler import SeededShuffleSampler
from emblema.pretraining.adapters.loaders.window_loader import WindowLoader
from emblema.shared.kernel.tokens import Token, TokenWindow
from tests.ml.onnx_export.dummy_set_encoder import DummySetEncoder

BUDGET = REPO_ROOT / "scripts" / "corpus_budget.toml"
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


def budget() -> dict[str, Any]:
    with BUDGET.open("rb") as handle:
        return tomllib.load(handle)


def default_window() -> WindowSpec:
    windows = budget()["corpora"][CORPUS]["windows"]
    spec = next(window for window in windows if window.get("default"))
    return WindowSpec(spec["length"], spec["stride"])


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
    return islice(produced, limit)


def synthetic_windows(count: int, window: WindowSpec) -> Iterator[TokenWindow]:
    """Windows of the shape the default C-MAPSS window has, for a machine without the raw data."""
    channels = budget()["corpora"][CORPUS]["channels"]
    length = int(window.length)
    for index in range(count):
        yield TokenWindow.of(
            Token(
                channel_id=1 + channel,
                value=float((index + step + channel) % 97) / 97.0,
                time=step / length,
                gap=1 / length,
            )
            for step in range(length)
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
    sampler = SeededShuffleSampler(size, seed=1)
    epoch = iter(range(1, 1000))

    def order() -> None:
        sampler.set_epoch(next(epoch))
        list(sampler)

    return median_seconds(order, 3)


def step_seconds(windows: Sequence[TokenWindow], *, width: int, layers: int, device: str) -> float:
    model = DummySetEncoder(d_model=width, n_heads=max(1, width // 64), n_layers=layers).to(device)
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


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [" | ".join(header), " | ".join("---" for _ in header)]
    lines += [" | ".join(row) for row in rows]
    return "\n".join(f"| {line} |" for line in lines)


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        # The headings use an em dash, which a Windows console's default code page lacks, and
        # the note this output is pasted into is LF-only.
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    device = device_in_use()
    corpus = load_corpus()
    tokens = corpus.median_tokens
    collating = {
        workers: collate_seconds(corpus.windows, num_workers=workers) for workers in (0, 2)
    }
    tiers = [tier for tier in budget()["tiers"] if tier["name"] in ("S", "M")]
    steps = {
        tier["name"]: step_seconds(
            corpus.windows, width=tier["width"], layers=tier["layers"], device=device
        )
        for tier in tiers
    }

    print(f"## {datetime.now():%Y-%m-%d} — {platform.system()} {platform.machine()}\n")
    print(
        table(
            ("", ""),
            [
                ("Machine", f"{platform.platform()}, {platform.processor() or 'unknown CPU'}"),
                ("Python", platform.python_version()),
                ("torch", version("torch")),
                ("numpy", version("numpy")),
                ("Device", device),
                ("Batch", str(BATCH_SIZE)),
            ],
        )
    )
    print("\n### Windows\n")
    print(
        table(
            ("Source", "Windows", "Tokens per window", "Held in memory per window"),
            [
                (
                    corpus.source,
                    str(len(corpus.windows)),
                    str(tokens),
                    f"{corpus.bytes_held / len(corpus.windows) / 1024:.1f} KiB",
                )
            ],
        )
    )
    print("\n### Collating\n")
    print(
        table(
            ("Workers", "s / batch", "Windows / s", "Tokens / s"),
            [
                (
                    str(workers),
                    f"{seconds * 1e3:.2f} ms",
                    f"{BATCH_SIZE / seconds:,.0f}",
                    f"{BATCH_SIZE * tokens / seconds:,.0f}",
                )
                for workers, seconds in collating.items()
            ],
        )
    )
    print("\n### Ordering an epoch\n")
    print(
        table(
            ("Windows in the corpus", "s / epoch"),
            [(f"{size:,}", f"{order_seconds(size) * 1e3:.1f} ms") for size in ORDERED_SIZES],
        )
    )
    print("\n### Training step, stand-in encoder\n")
    print(
        table(
            ("Tier", "Width", "Layers", "s / step", "Windows / s"),
            [
                (
                    tier["name"],
                    str(tier["width"]),
                    str(tier["layers"]),
                    f"{steps[tier['name']] * 1e3:.0f} ms",
                    f"{BATCH_SIZE / steps[tier['name']]:,.0f}",
                )
                for tier in tiers
            ],
        )
    )
    print("\n### Verdict\n")
    for name, seconds in steps.items():
        for workers, collated in collating.items():
            print(
                f"- Tier {name}, {workers} worker(s): a step costs "
                f"{seconds / collated:.1f}x collating the batch it consumes."
            )


if __name__ == "__main__":
    main()
