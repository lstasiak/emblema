"""Register a raw corpus, tokenise it and publish the result as an artifact.

Preprocessing is done once, on a machine that holds the raw data and has no session to run out
of; a training run then mounts what this produced and starts training without parsing a source
file. Run it as::

    uv run python -m emblema.entrypoints.cli.publish_corpus
        --corpus cmapss --root data/raw/cmapss --window 50 --stride 5

What it prints is the reference to the manifest — key and checksum — which is what a run needs
to find the corpus and all it needs to decide whether that corpus is the one it wants.
"""

import argparse
from pathlib import Path

from emblema.catalog.adapters.readers.cmapss import SUBSETS
from emblema.catalog.application.register_corpus import RegisterCorpusCommand
from emblema.catalog.application.register_corpus_version import RegisterCorpusVersionCommand
from emblema.catalog.application.tokenise_corpus_version import TokeniseCorpusVersionCommand
from emblema.catalog.domain.corpus_source import CorpusSource
from emblema.catalog.domain.licence import Licence
from emblema.catalog.domain.window_spec import WindowSpec
from emblema.config.settings import Settings
from emblema.entrypoints.cli.composition import Services, build_services
from emblema.shared.kernel.artifacts import ArtifactRef

# What is known about the corpora this process can publish, beyond how to read them: facts about
# the source rather than about the bytes, which no reader can state.
SOURCES = {
    "cmapss": (
        CorpusSource("NASA Prognostics Center of Excellence", "https://data.nasa.gov/dataset/"),
        Licence("US Government Work", permits_derivatives=False),
    )
}


def publish(services: Services, arguments: argparse.Namespace) -> ArtifactRef:
    """Register the corpus, freeze what the reader sees as a version, and publish it tokenised."""
    source, licence = SOURCES[arguments.corpus]
    corpus_id = services.register_corpus(
        RegisterCorpusCommand(name=arguments.corpus, source=source)
    )
    version = services.register_corpus_version(
        RegisterCorpusVersionCommand(corpus_id=corpus_id, licence=licence)
    )
    return services.tokenise_corpus_version(
        TokeniseCorpusVersionCommand(
            corpus_id=corpus_id,
            version_id=version.version_id,
            window=WindowSpec(arguments.window, arguments.stride),
            validation_fraction=arguments.validation_fraction,
            seed=arguments.seed,
        )
    )


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a tokenised corpus as an artifact.")
    parser.add_argument("--corpus", choices=sorted(SOURCES), required=True)
    parser.add_argument("--root", type=Path, required=True, help="directory of the raw corpus")
    parser.add_argument("--subset", action="append", help="subset to read; repeatable")
    parser.add_argument("--window", type=float, required=True, help="window length, in time")
    parser.add_argument("--stride", type=float, required=True, help="stride between windows")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("data/artifacts"),
        help="where blocks pass through on their way to the store",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - reads the environment
    arguments = parse(argv)
    settings = Settings()
    services = build_services(
        settings,
        corpus_root=arguments.root,
        workspace=arguments.workspace,
        subsets=tuple(arguments.subset) if arguments.subset else SUBSETS,
    )
    ref = publish(services, arguments)
    print(f"{ref.key}\n{ref.checksum.algorithm}:{ref.checksum.digest}")


if __name__ == "__main__":
    main()
