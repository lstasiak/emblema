import argparse
from collections.abc import Sequence
from pathlib import Path

from emblema.catalog.application.use_cases.publish_corpus import PublishCorpusCommand
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.tokenisation.split_policy import NamedSplit, SeededSplit, SplitPolicy
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.config.settings import Settings
from emblema.entrypoints.cli.publish_corpus.composition_root import CompositionRoot
from emblema.entrypoints.cli.publish_corpus.known_corpora import KnownCorpora
from emblema.entrypoints.cli.publish_corpus.publish_corpus_invocation import PublishCorpusInvocation
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum

# Where a downloaded corpus is unpacked, one directory per corpus. Stated as a default rather
# than demanded on every run: a corpus that is generated has nothing to unpack.
RAW = Path("data") / "raw"


class PublishCorpusCli:
    """Command line that registers a raw corpus, tokenises it and publishes the artifact.

    Prints the reference to the manifest, key and checksum, which is all a run needs to find the
    corpus and decide whether it is the one it wants. Run as::

        uv run python -m emblema.entrypoints.cli.publish_corpus
            --corpus cmapss --window 50 --stride 5

    A corpus to be trained on together with an earlier one continues that one's vocabulary::

        ... --vocabulary-from <manifest key> <manifest checksum>
    """

    def __init__(self, known: KnownCorpora | None = None) -> None:
        self._known = KnownCorpora.default() if known is None else known

    def parse(self, argv: Sequence[str] | None = None) -> PublishCorpusInvocation:
        arguments = self._parser().parse_args(argv)
        known = self._known.named(arguments.corpus)
        return PublishCorpusInvocation(
            command=PublishCorpusCommand(
                name=known.name,
                source=known.source,
                licence=known.licence,
                window=WindowSpec(arguments.window, arguments.stride),
                split=self._split(arguments),
                vocabulary_from=self._vocabulary_from(arguments.vocabulary_from),
            ),
            corpus_root=arguments.root or RAW / known.name,
            workspace=arguments.workspace,
            subsets=tuple(arguments.subset or ()),
        )

    @staticmethod
    def _split(arguments: argparse.Namespace) -> SplitPolicy:
        """What the command line asked for: units named, or a share drawn by the seed."""
        if arguments.hold_out:
            return NamedSplit.of(UnitKey(key) for key in arguments.hold_out)
        return SeededSplit(arguments.validation_fraction, arguments.seed)

    def run(self, argv: Sequence[str] | None = None) -> None:  # pragma: no cover - environment
        invocation = self.parse(argv)
        root = CompositionRoot(
            Settings(),
            corpus=invocation.command.name,
            corpus_root=invocation.corpus_root,
            workspace=invocation.workspace,
            subsets=invocation.subsets,
        )
        ref = root.services.publish_corpus(invocation.command)
        print(f"{ref.key}\n{ref.checksum}")

    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Publish a tokenised corpus as an artifact.")
        parser.add_argument("--corpus", choices=self._known.names(), required=True)
        parser.add_argument(
            "--root",
            type=Path,
            help=f"directory of the raw corpus; {RAW}/<corpus> unless given, and unused by a "
            "corpus that is generated rather than downloaded",
        )
        parser.add_argument(
            "--subset", action="append", help="subset to read; repeatable, all of them unless given"
        )
        parser.add_argument("--window", type=float, required=True, help="window length, in time")
        parser.add_argument("--stride", type=float, required=True, help="stride between windows")
        parser.add_argument("--validation-fraction", type=float, default=0.2)
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument(
            "--hold-out",
            nargs="+",
            metavar="UNIT",
            help="units to hold out by name, for a corpus whose units differ in kind; the "
            "fraction and the seed are not read when this is given",
        )
        parser.add_argument(
            "--vocabulary-from",
            nargs=2,
            metavar=("KEY", "CHECKSUM"),
            help="manifest of an earlier corpus whose channel vocabulary this one continues",
        )
        parser.add_argument(
            "--workspace",
            type=Path,
            default=Path("data/artifacts"),
            help="where blocks pass through on their way to the store",
        )
        return parser

    @staticmethod
    def _vocabulary_from(pair: Sequence[str] | None) -> ArtifactRef | None:
        if not pair:
            return None
        key, checksum = pair
        return ArtifactRef(key, Checksum.parse(checksum))
