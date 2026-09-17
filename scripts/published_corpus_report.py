"""Publish a corpus the way a run will, and report what the artifact costs and what it holds.

Answered once, as preprocessing happens once: the artifact's size beside the same windows as Python
objects, whether publishing twice into one registry yields one artifact, and what reading a window
from the memory map costs against a list. The artifacts go to a local directory store: the numbers
concern the format and the machine, not a network.

    uv sync --all-extras
    uv run scripts/published_corpus_report.py --corpus cmapss

Run where the raw corpus is; prints markdown for the verification note.
"""

import argparse
import statistics
import sys
import time
import tomllib
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Run from anywhere: the reporting helpers live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.application.use_cases.publish_corpus import PublishCorpusCommand
from emblema.catalog.domain.tokenisation.split_policy import SeededSplit
from emblema.catalog.domain.tokenisation.tokenisation_manifest import TokenisationManifest
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.catalog.ports.corpus_archive import CorpusArchive
from emblema.catalog.ports.corpus_repository import CorpusRepository
from emblema.entrypoints.cli.publish_corpus.composition_root import CompositionRoot
from emblema.entrypoints.cli.publish_corpus.known_corpora import KnownCorpora
from emblema.shared.adapters.storage.local_directory import LocalDirectoryArtifactStore
from emblema.shared.adapters.windows.window_block_writer import WindowBlockWriter
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow
from scripts.raw_corpora import raw_root
from scripts.reporting import dated_heading, machine, table

BUDGET = REPO_ROOT / "scripts" / "corpus_budget.toml"
RAW = REPO_ROOT / "data" / "raw"
# How many windows the read-back timings are taken over. Enough for a median to mean something,
# few enough that holding them as objects beside the map is not itself the measurement.
SAMPLED_WINDOWS = 512


@dataclass(frozen=True)
class Published:
    """One publication: what went in, what came out, where it can be read and how long it took."""

    manifest: TokenisationManifest
    manifest_ref: ArtifactRef
    archive: CorpusArchive
    block_bytes: int
    manifest_bytes: int
    seconds: float


@dataclass(frozen=True)
class Writing:
    """What putting one window into a block costs, and how much of that is checking the window.

    The check is the same one a reader pays, so it is also the yardstick that tells a slower
    machine from a slower code path between two runs.
    """

    add_seconds: float
    check_seconds: float
    windows: int

    @property
    def add_ms(self) -> float:
        return self.add_seconds * 1e3

    @property
    def check_ms(self) -> float:
        return self.check_seconds * 1e3


@dataclass(frozen=True)
class ReadBack:
    """What reading windows out of the artifact costs, and what holding them instead would."""

    seconds: float
    bytes_as_objects: int
    windows: int

    @property
    def per_window_ms(self) -> float:
        return self.seconds * 1e3 / max(self.windows, 1)

    @property
    def per_window_bytes(self) -> float:
        return self.bytes_as_objects / max(self.windows, 1)


def default_window(corpus: str) -> WindowSpec:
    with BUDGET.open("rb") as handle:
        windows = tomllib.load(handle)["corpora"][corpus]["windows"]
    spec = next(window for window in windows if window.get("default"))
    return WindowSpec(float(spec["length"]), float(spec["stride"]))


def measured_counts(corpus: str) -> dict[str, int] | None:
    """What the data spike counted for this corpus, when it has run on this machine."""
    with BUDGET.open("rb") as handle:
        entry: dict[str, Any] = tomllib.load(handle)["corpora"][corpus]
    measured = entry.get("measured")
    return dict(measured) if measured else None


def corpus_root(corpus: str) -> Path:
    root = raw_root(corpus)
    if root is None:
        raise SystemExit(f"no raw {corpus} under {RAW / corpus}; this report needs the real corpus")
    return root


def publish_once(
    root: Path,
    workspace: Path,
    subsets: Sequence[str] | None,
    command: PublishCorpusCommand,
    registry: CorpusRepository,
) -> Published:
    store = LocalDirectoryArtifactStore(workspace / "store")
    process = CompositionRoot.over(
        corpus=command.name,
        corpus_root=root,
        workspace=workspace / "blocks",
        subsets=tuple(subsets or ()),
        corpora=registry,
        store=store,
    )
    started = time.perf_counter()
    ref = process.services.publish_corpus(command)
    seconds = time.perf_counter() - started
    archive = process.adapters.archive
    manifest = archive.read_manifest(ref)
    return Published(
        manifest=manifest,
        manifest_ref=ref,
        archive=archive,
        # The block just published is left in the workspace under its digest, for its reader.
        block_bytes=(workspace / "blocks" / manifest.block.checksum.digest).stat().st_size,
        manifest_bytes=len(store.get(ref)),
        seconds=seconds,
    )


def read_back(published: Published) -> ReadBack:
    """Read windows out of the published artifact, and weigh what holding them would cost."""
    mapped = published.archive.read_windows(published.manifest.archived, published.manifest.units)
    count = min(SAMPLED_WINDOWS, len(mapped))
    seconds = _median_seconds(lambda: [mapped[index] for index in range(count)])
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    held: list[TokenWindow] = [mapped[index] for index in range(count)]
    bytes_held = tracemalloc.get_traced_memory()[0] - before
    tracemalloc.stop()
    del held
    return ReadBack(seconds, bytes_held, count)


def writing(published: Published, workspace: Path) -> Writing:
    """Time the writer on windows of the published corpus, cast and checked as it writes them."""
    mapped = published.archive.read_windows(published.manifest.archived, published.manifest.units)
    count = min(SAMPLED_WINDOWS, len(mapped))
    windows: list[TokenWindow] = [mapped[index] for index in range(count)]
    with WindowBlockWriter(workspace / "timing" / "sample.block") as writer:

        def add_all() -> None:
            for window in windows:
                writer.add(window, unit=0, start=0.0, end=1.0)

        add_seconds = _median_seconds(add_all)
    check_seconds = _median_seconds(
        lambda: [
            TokenWindow(
                channel_ids=window.channel_ids,
                values=window.values,
                times=window.times,
                gaps=window.gaps,
                timeless=window.timeless,
            )
            for window in windows
        ]
    )
    return Writing(add_seconds / count, check_seconds / count, count)


def _median_seconds(step: Callable[[], object], repeats: int = 5) -> float:
    """The median of ``repeats`` timings, the first one thrown away as warm-up."""
    timings = []
    for _ in range(repeats + 1):
        started = time.perf_counter()
        step()
        timings.append(time.perf_counter() - started)
    return statistics.median(timings[1:])


def render(
    corpus: str,
    first: Published,
    second: Published,
    reading: ReadBack,
    writes: Writing,
    window: WindowSpec,
) -> str:
    manifest = first.manifest
    spike = measured_counts(corpus)
    per_window = first.block_bytes / max(manifest.window_count, 1)
    rows = [
        ("Machine", machine()),
        ("Corpus", f"{corpus}, window length {window.length:g} stride {window.stride:g}"),
        ("Units", f"{len(manifest.units)} indexed, {len(manifest.empty_units)} yielding no window"),
        (
            "Split",
            f"{len(manifest.split.training)} training, "
            f"{len(manifest.split.validation)} validation, seed {manifest.split_seed}",
        ),
        ("Windows", f"{manifest.window_count:,}"),
        ("Tokens", f"{manifest.token_count:,}"),
        ("Block", f"{first.block_bytes / 1e6:,.1f} MB ({per_window / 1024:.1f} KiB per window)"),
        ("Manifest", f"{first.manifest_bytes / 1024:,.1f} KiB"),
        ("Publishing", f"{first.seconds:,.1f} s"),
    ]
    if spike:
        rows.append(
            ("Data spike", f"{spike['units']} units, {spike['observations']:,} observations")
        )
    same_block = first.manifest.block == second.manifest.block
    same_manifest = first.manifest_ref == second.manifest_ref
    repeated = [
        "Publishing the same corpus twice gives "
        + ("one block" if same_block else "**two blocks**")
        + " and "
        + ("one manifest." if same_manifest else "**two manifests**.")
    ]
    if same_block and not same_manifest:
        repeated.append(
            "The blocks agree, so the data is one artifact. The manifests differ, so the second "
            "run did not find the first run's registration and minted a new version identifier."
        )
    lines = [dated_heading(), "", table(("", ""), rows), ""]
    lines += [
        *repeated,
        "",
        table(
            ("Reading back", f"over {reading.windows} windows"),
            [
                ("One window from the map", f"{reading.per_window_ms:,.2f} ms"),
                ("A batch of eight", f"{reading.per_window_ms * 8:,.1f} ms"),
                (
                    "Held as objects",
                    f"{reading.per_window_bytes / 1024:,.1f} KiB per window, "
                    f"{reading.per_window_bytes * manifest.window_count / 1e9:,.2f} GB in all",
                ),
                (
                    "On disk",
                    f"{per_window / 1024:,.1f} KiB per window, "
                    f"{first.block_bytes / 1e9:,.2f} GB in all",
                ),
            ],
        ),
        "",
        table(
            ("Writing", f"over {writes.windows} windows"),
            [
                ("One window into the block", f"{writes.add_ms:,.2f} ms"),
                ("Of which checking it reads back as a window", f"{writes.check_ms:,.2f} ms"),
                (
                    "Checking alone, over the whole corpus",
                    f"{writes.check_seconds * manifest.window_count:,.1f} s",
                ),
            ],
        ),
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="cmapss", choices=KnownCorpora.default().names())
    parser.add_argument("--subset", action="append")
    parser.add_argument("--workspace", type=Path, default=REPO_ROOT / "data" / "report")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args(argv)
    root = corpus_root(arguments.corpus)
    window = default_window(arguments.corpus)
    known = KnownCorpora.default().named(arguments.corpus)
    command = PublishCorpusCommand(
        name=known.name,
        source=known.source,
        licence=known.licence,
        window=window,
        split=SeededSplit(arguments.validation_fraction, arguments.seed),
    )
    # One registry for both publications, in memory: the report is about the format and the
    # machine, so it stands in for the database the process would otherwise register into.
    registry = InMemoryCorpusRepository()
    first = publish_once(root, arguments.workspace / "first", arguments.subset, command, registry)
    second = publish_once(root, arguments.workspace / "second", arguments.subset, command, registry)
    reading = read_back(first)
    writes = writing(first, arguments.workspace / "first")
    print(render(arguments.corpus, first, second, reading, writes, window))


if __name__ == "__main__":
    main()
