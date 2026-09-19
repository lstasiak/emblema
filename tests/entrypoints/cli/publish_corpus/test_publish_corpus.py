"""The publishing process assembled and run end to end, on adapters that reach nothing outside.

What is checked is the wiring: that the process the composition root builds publishes a corpus,
and that every use case received the adapter the process was given rather than a default.
"""

from pathlib import Path
from typing import NamedTuple

import pytest

from emblema.catalog.adapters.archive.block_corpus_archive import BlockCorpusArchive
from emblema.catalog.adapters.in_memory.corpus_reader import InMemoryCorpusReader
from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.adapters.persistence.corpus_repository import SqlAlchemyCorpusRepository
from emblema.catalog.adapters.readers.cmapss import CmapssCorpusReader
from emblema.catalog.adapters.synthetic.synthetic_corpus_reader import SyntheticCorpusReader
from emblema.catalog.application.use_cases.publish_corpus import PublishCorpusCommand
from emblema.catalog.domain.exceptions import InvalidUnitSplitError
from emblema.catalog.domain.tokenisation.split_policy import SeededSplit, SubsetSplit
from emblema.entrypoints.cli.publish_corpus.composition_root import CompositionRoot
from emblema.entrypoints.cli.publish_corpus.known_corpora import KnownCorpora
from emblema.entrypoints.cli.publish_corpus.publish_corpus_cli import (
    SEED,
    VALIDATION_FRACTION,
    PublishCorpusCli,
)
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.storage.s3 import S3ArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.catalog.domain.support import SCHEMA, description, measured_units
from tests.support.settings import unreachable_store

DATA = b"records"
UNITS = measured_units(4)
ARGUMENTS = ["--corpus", "cmapss", "--root", "data/raw/cmapss", "--window", "4", "--stride", "2"]


class Process(NamedTuple):
    """The assembled process and the store its artifacts were meant to land in."""

    root: CompositionRoot
    store: InMemoryArtifactStore


@pytest.fixture
def process(tmp_path: Path) -> Process:
    told = description(DATA, SCHEMA, units=len(UNITS), observations=sum(len(o) for _, o in UNITS))
    store = InMemoryArtifactStore()
    root = CompositionRoot(
        unreachable_store(),
        corpus_root=tmp_path / "raw",
        workspace=tmp_path / "workspace",
        corpora=InMemoryCorpusRepository(),
        reader=InMemoryCorpusReader(told, UNITS),
        store=store,
    )
    return Process(root, store)


def command(*extra: str) -> PublishCorpusCommand:
    return PublishCorpusCli().parse([*ARGUMENTS, *extra]).command


def test_the_process_publishes_a_corpus_that_reads_back(process: Process) -> None:
    ref = process.root.services.publish_corpus(command("--validation-fraction", "0.25"))

    manifest = process.root.adapters.archive.read_manifest(ref)
    windows = process.root.adapters.archive.read_windows(manifest.archived, manifest.split.training)
    assert manifest.corpus == "cmapss"
    assert len(windows) > 0


def test_the_artifacts_land_in_the_store_the_process_was_given(process: Process) -> None:
    # Every use case holding its own store would leave each of them correct and the process
    # broken; the corpus would be published where nobody looks for it.
    ref = process.root.services.publish_corpus(command("--validation-fraction", "0.25"))

    manifest = process.root.adapters.archive.read_manifest(ref)
    assert process.store.exists(ref)
    assert process.store.exists(manifest.block)


def test_publishing_twice_through_one_process_yields_one_manifest(process: Process) -> None:
    # The use cases share one repository: were each holding its own, the second publication
    # would register the corpus again and mint a second manifest for the same data.
    first = process.root.services.publish_corpus(command("--validation-fraction", "0.25"))
    second = process.root.services.publish_corpus(command("--validation-fraction", "0.25"))

    assert first == second


def test_what_the_command_line_says_is_what_the_corpus_was_cut_with(process: Process) -> None:
    ref = process.root.services.publish_corpus(
        command("--validation-fraction", "0.25", "--seed", "7", "--window", "2")
    )

    manifest = process.root.adapters.archive.read_manifest(ref)
    assert (manifest.window.length, manifest.window.stride) == (2.0, 2.0)
    assert manifest.split_seed == 7
    assert len(manifest.split.validation) == 1


def test_the_command_line_holds_out_the_units_it_names(process: Process) -> None:
    """Naming the units is how a corpus whose units differ in kind states its held-out side."""
    units = process.root.adapters.reader.read_units()
    named = next(iter(units)).key

    ref = process.root.services.publish_corpus(command("--hold-out", named.value))

    manifest = process.root.adapters.archive.read_manifest(ref)
    assert manifest.split.validation == frozenset({named})
    assert manifest.split_seed is None


def test_a_unit_the_corpus_does_not_hold_stops_the_publication(process: Process) -> None:
    with pytest.raises(InvalidUnitSplitError, match="does not hold"):
        process.root.services.publish_corpus(command("--hold-out", "no-such-unit"))


def test_without_overrides_the_process_runs_on_what_the_settings_name(tmp_path: Path) -> None:
    # The overrides above are what every other test here uses, so nothing would otherwise exercise
    # the wiring a real run gets: the bucket and the database named by the environment, behind a
    # block archive, over the reader of the corpus asked for.
    root = CompositionRoot(
        unreachable_store(),
        corpus="cmapss",
        corpus_root=tmp_path / "raw",
        workspace=tmp_path / "workspace",
    )

    assert isinstance(root.adapters.store, S3ArtifactStore)
    assert isinstance(root.adapters.corpora, SqlAlchemyCorpusRepository)
    assert isinstance(root.adapters.archive, BlockCorpusArchive)
    assert isinstance(root.adapters.reader, CmapssCorpusReader)


def test_the_corpus_asked_for_decides_which_adapter_reads_it(tmp_path: Path) -> None:
    root = CompositionRoot(
        unreachable_store(),
        corpus="control-a",
        corpus_root=tmp_path / "raw",
        workspace=tmp_path / "workspace",
    )

    assert isinstance(root.adapters.reader, SyntheticCorpusReader)


@pytest.mark.parametrize("corpus", KnownCorpora.default().names())
def test_every_corpus_the_command_line_offers_has_an_adapter_that_reads_it(
    tmp_path: Path, corpus: str
) -> None:
    # The names the parser accepts and the names the root has a reader for are two lists kept by
    # hand. A corpus on the first and missing from the second is refused only once a run has been
    # started, with the raw data already fetched and named on the command line.
    root = CompositionRoot.over(
        corpora=InMemoryCorpusRepository(),
        store=InMemoryArtifactStore(),
        corpus=corpus,
        corpus_root=tmp_path / "raw",
        workspace=tmp_path / "workspace",
    )

    assert type(root.adapters.reader).__module__.startswith("emblema.catalog.adapters")


def test_a_corpus_no_adapter_reads_is_refused_when_the_process_is_assembled(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no adapter reads"):
        CompositionRoot(
            unreachable_store(),
            corpus="nothing-of-the-sort",
            corpus_root=tmp_path / "raw",
            workspace=tmp_path / "workspace",
        )


def test_a_process_given_neither_a_corpus_nor_a_reader_is_refused(tmp_path: Path) -> None:
    # Naming a corpus is how the root chooses a reader, so a process that neither names one nor
    # brings its own has no reader to run on and says so here rather than later.
    with pytest.raises(ValueError, match="needs a corpus to read"):
        CompositionRoot(
            unreachable_store(),
            corpus_root=tmp_path / "raw",
            workspace=tmp_path / "workspace",
        )


def test_a_process_given_every_adapter_it_would_build_from_settings_reads_none(
    tmp_path: Path,
) -> None:
    store = InMemoryArtifactStore()

    root = CompositionRoot(
        corpus="control-a",
        corpus_root=tmp_path / "raw",
        workspace=tmp_path / "workspace",
        corpora=InMemoryCorpusRepository(),
        store=store,
    )

    assert root.adapters.store is store


def test_a_process_left_to_build_an_adapter_without_settings_is_refused(tmp_path: Path) -> None:
    raw, workspace = tmp_path / "raw", tmp_path / "workspace"

    with pytest.raises(ValueError, match="needs store given"):
        CompositionRoot(
            corpus="control-a",
            corpus_root=raw,
            workspace=workspace,
            corpora=InMemoryCorpusRepository(),
        )
    with pytest.raises(ValueError, match="needs corpora given"):
        CompositionRoot(
            corpus="control-a", corpus_root=raw, workspace=workspace, store=InMemoryArtifactStore()
        )


def test_the_command_line_states_the_window_it_was_given() -> None:
    invocation = PublishCorpusCli().parse(ARGUMENTS)

    assert (invocation.command.window.length, invocation.command.window.stride) == (4.0, 2.0)
    assert invocation.command.split == SeededSplit(0.2, 1)
    assert invocation.command.vocabulary_from is None
    assert invocation.corpus_root == Path("data/raw/cmapss")
    assert invocation.subsets == ()


def test_a_corpus_is_looked_for_under_its_own_name_unless_a_root_is_given() -> None:
    invocation = PublishCorpusCli().parse(
        ["--corpus", "control-a", "--window", "4", "--stride", "2"]
    )

    assert invocation.corpus_root == Path("data/raw/control-a")


def test_the_subsets_asked_for_are_the_ones_named() -> None:
    invocation = PublishCorpusCli().parse([*ARGUMENTS, "--subset", "FD002"])

    assert invocation.subsets == ("FD002",)


def test_the_command_line_names_the_manifest_whose_vocabulary_to_continue() -> None:
    digest = "0" * 64
    parsed = command("--vocabulary-from", f"durable/sha256/{digest}", f"sha256:{digest}")

    assert parsed.vocabulary_from == ArtifactRef(
        f"durable/sha256/{digest}", Checksum.parse(f"sha256:{digest}")
    )


def test_a_subset_asked_for_is_the_only_one_read() -> None:
    invocation = PublishCorpusCli().parse([*ARGUMENTS, "--subset", "FD001"])

    assert invocation.subsets == ("FD001",)


def test_naming_the_units_and_drawing_them_at_once_is_refused() -> None:
    """A fraction or a seed beside named units draws nothing, so it is a contradiction."""
    for contradiction in (["--seed", "7"], ["--validation-fraction", "0.3"]):
        with pytest.raises(SystemExit):
            PublishCorpusCli().parse([*ARGUMENTS, "--hold-out", "FD001_unit_1", *contradiction])
        with pytest.raises(SystemExit):
            PublishCorpusCli().parse([*ARGUMENTS, "--hold-out-subset", "FD001", *contradiction])


def test_the_command_line_holds_out_the_subset_it_was_given() -> None:
    invocation = PublishCorpusCli().parse([*ARGUMENTS, "--hold-out-subset", "FD002"])

    assert invocation.command.split == SubsetSplit("FD002")


def test_naming_the_units_and_holding_out_a_subset_at_once_is_refused() -> None:
    """Both say which units are held out, and nothing says which of them the run meant."""
    with pytest.raises(SystemExit):
        PublishCorpusCli().parse(
            [*ARGUMENTS, "--hold-out", "FD001/1", "--hold-out-subset", "FD002"]
        )


def test_holding_out_a_subset_the_run_does_not_read_is_refused_before_the_corpus_is() -> None:
    """A publication of a large corpus reads for minutes; this contradiction is stated up front."""
    with pytest.raises(SystemExit):
        PublishCorpusCli().parse([*ARGUMENTS, "--subset", "FD001", "--hold-out-subset", "FD002"])


def test_a_command_line_that_says_nothing_of_the_split_draws_the_usual_share() -> None:
    assert PublishCorpusCli().parse(ARGUMENTS).command.split == SeededSplit(
        VALIDATION_FRACTION, SEED
    )
