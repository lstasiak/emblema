import argparse
from collections.abc import Sequence
from pathlib import Path

from emblema.catalog.application.use_cases.publish_corpus import PublishCorpusCommand
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.tokenisation.split_policy import (
    NamedSplit,
    SeededSplit,
    SplitPolicy,
    SubsetSplit,
)
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
# What a publication holds out where the command line says nothing: a fifth of the units, drawn
# by the seed every publication of this project has used.
VALIDATION_FRACTION = 0.2
SEED = 1


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
        parser = self._parser()
        arguments = parser.parse_args(argv)
        if arguments.hold_out and arguments.hold_out_subset:
            parser.error(
                "--hold-out and --hold-out-subset both say which units are held out; give one"
            )
        if (arguments.hold_out or arguments.hold_out_subset) and (
            arguments.validation_fraction is not None or arguments.seed is not None
        ):
            parser.error(
                "--hold-out and --hold-out-subset say which units are held out, so there is "
                "nothing for --validation-fraction or --seed to draw"
            )
        if (
            arguments.hold_out_subset
            and arguments.subset
            and arguments.hold_out_subset not in arguments.subset
        ):
            # Refused here rather than where the split is made: the corpus is read first, and a
            # publication of a large one would fail minutes in on a contradiction stated up front.
            parser.error(
                f"--hold-out-subset {arguments.hold_out_subset} holds out a subset this run does "
                "not read"
            )
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
        """What the command line asked for: units named, a subset held out, or a seeded share."""
        if arguments.hold_out:
            return NamedSplit.of(UnitKey(key) for key in arguments.hold_out)
        if arguments.hold_out_subset:
            return SubsetSplit(arguments.hold_out_subset)
        return SeededSplit(
            VALIDATION_FRACTION
            if arguments.validation_fraction is None
            else arguments.validation_fraction,
            SEED if arguments.seed is None else arguments.seed,
        )

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
        # These default to nothing rather than to their values, so that a fraction or a seed
        # given beside named units is a contradiction the command line can refuse rather than
        # quietly drop.
        parser.add_argument(
            "--validation-fraction",
            type=float,
            default=None,
            help=f"share of units held out from fitting; {VALIDATION_FRACTION} unless given",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help=f"seed the split is drawn with; {SEED} unless given",
        )
        parser.add_argument(
            "--hold-out",
            nargs="+",
            metavar="UNIT",
            help="units to hold out by name, for a corpus whose units differ in kind; refused "
            "together with --validation-fraction or --seed, which then draw nothing",
        )
        parser.add_argument(
            "--hold-out-subset",
            metavar="SUBSET",
            help="subset whose every unit is held out, where the corpus's publisher drew the "
            "line and its units are too many to name",
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
