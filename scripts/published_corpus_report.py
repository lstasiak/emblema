"""Publish a corpus the way a run will, and report what the artifact costs and what it holds.

Preprocessing happens once and every later run reads the result, so the questions worth answering
about it are answered once too: how large the artifact is beside the same windows held as Python
objects, whether publishing twice gives one artifact or two, and what reading a window back out of
a memory map costs against reading one from a list. Run it where the raw corpus is:

    uv sync --all-extras
    uv run scripts/published_corpus_report.py --corpus cmapss

The artifacts go to a local directory store rather than the bucket: the numbers here are about the
format and the machine, not about a network. The output is markdown, meant to be pasted under a
dated heading in the verification note.
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

from emblema.catalog.adapters.archive.block_corpus_archive import BlockCorpusArchive
from emblema.catalog.adapters.archive.manifest_json import encode_manifest
from emblema.catalog.domain.tokenisation_manifest import TokenisationManifest
from emblema.config.artifact_store_settings import ArtifactStoreSettings
from emblema.config.settings import Settings
from emblema.entrypoints.cli.composition import build_services
from emblema.entrypoints.cli.publish_corpus import parse as parse_publish
from emblema.entrypoints.cli.publish_corpus import publish
from emblema.shared.adapters.storage.local_directory import LocalDirectoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow
from scripts.reporting import dated_heading, machine, table

BUDGET = REPO_ROOT / "scripts" / "corpus_budget.toml"
RAW = REPO_ROOT / "data" / "raw"
# How many windows the read-back timings are taken over. Enough for a median to mean something,
# few enough that holding them as objects beside the map is not itself the measurement.
SAMPLED_WINDOWS = 512


@dataclass(frozen=True)
class Published:
    """One publication: what went in, what came out and how long it took."""

    manifest: TokenisationManifest
    manifest_ref: ArtifactRef
    block_bytes: int
    manifest_bytes: int
    seconds: float


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


def default_window(corpus: str) -> tuple[float, float]:
    with BUDGET.open("rb") as handle:
        windows = tomllib.load(handle)["corpora"][corpus]["windows"]
    spec = next(window for window in windows if window.get("default"))
    return float(spec["length"]), float(spec["stride"])


def measured_counts(corpus: str) -> dict[str, int] | None:
    """What the data spike counted for this corpus, when it has run on this machine."""
    with BUDGET.open("rb") as handle:
        entry: dict[str, Any] = tomllib.load(handle)["corpora"][corpus]
    measured = entry.get("measured")
    return dict(measured) if measured else None


def corpus_root(corpus: str) -> Path:
    hits = sorted((RAW / corpus).rglob("*.txt")) if (RAW / corpus).is_dir() else []
    if not hits:
        raise SystemExit(f"no raw {corpus} under {RAW / corpus}; this report needs the real corpus")
    return hits[0].parent


def settings() -> Settings:
    """Settings with a store that is never reached: the archive below is given its own."""
    return Settings(
        artifact_store=ArtifactStoreSettings(
            endpoint_url="http://127.0.0.1:3900",
            region="garage",
            bucket="emblema",
            key_prefix="dev",
        )
    )


def publish_once(
    corpus: str, root: Path, workspace: Path, arguments: argparse.Namespace
) -> Published:
    store = LocalDirectoryArtifactStore(workspace / "store")
    archive = BlockCorpusArchive(store, workspace / "blocks")
    services = build_services(
        settings(), corpus_root=root, workspace=workspace / "blocks", archive=archive
    )
    started = time.perf_counter()
    ref = publish(services, arguments)
    seconds = time.perf_counter() - started
    manifest = archive.read_manifest(ref)
    return Published(
        manifest=manifest,
        manifest_ref=ref,
        block_bytes=(workspace / "store").joinpath(*manifest.block.key.split("/")).stat().st_size,
        manifest_bytes=len(encode_manifest(manifest)),
        seconds=seconds,
    )


def read_back(published: Published, workspace: Path) -> ReadBack:
    """Read windows out of the published artifact, and weigh what holding them would cost."""
    store = LocalDirectoryArtifactStore(workspace / "store")
    archive = BlockCorpusArchive(store, workspace / "blocks")
    mapped = archive.read_windows(published.manifest, published.manifest.units)
    count = min(SAMPLED_WINDOWS, len(mapped))
    seconds = _median_seconds(lambda: [mapped[index] for index in range(count)])
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    held: list[TokenWindow] = [mapped[index] for index in range(count)]
    bytes_held = tracemalloc.get_traced_memory()[0] - before
    tracemalloc.stop()
    del held
    return ReadBack(seconds, bytes_held, count)


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
    window: tuple[float, float],
) -> str:
    manifest = first.manifest
    spike = measured_counts(corpus)
    per_window = first.block_bytes / max(manifest.window_count, 1)
    rows = [
        ("Machine", machine()),
        ("Corpus", f"{corpus}, window length {window[0]:g} stride {window[1]:g}"),
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
            "The blocks agree, so the data is one artifact. The manifests differ because each run "
            "registers the corpus afresh and mints a new version identifier: no registration "
            "outlives the process yet. Once one does, the second run finds the version already "
            "frozen and describes it rather than minting another."
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
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="cmapss")
    parser.add_argument("--subset", action="append")
    parser.add_argument("--workspace", type=Path, default=REPO_ROOT / "data" / "report")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args(argv)
    root = corpus_root(arguments.corpus)
    window = default_window(arguments.corpus)
    publish_arguments = parse_publish(
        [
            "--corpus",
            arguments.corpus,
            "--root",
            str(root),
            "--window",
            str(window[0]),
            "--stride",
            str(window[1]),
            "--validation-fraction",
            str(arguments.validation_fraction),
            "--seed",
            str(arguments.seed),
            *(argument for subset in arguments.subset or [] for argument in ("--subset", subset)),
        ]
    )
    first = publish_once(arguments.corpus, root, arguments.workspace / "first", publish_arguments)
    second = publish_once(arguments.corpus, root, arguments.workspace / "second", publish_arguments)
    reading = read_back(first, arguments.workspace / "first")
    print(render(arguments.corpus, first, second, reading, window))


if __name__ == "__main__":
    main()
