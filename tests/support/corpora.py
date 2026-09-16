"""The corpora the tests read, and what it takes to publish one.

Each downloaded corpus comes in two sizes: the miniature sample committed to the repository, which
every machine has, and the full download, which only a machine that fetched it has. Stated once,
because a test that anchors the sample on its own depth in the tree breaks when it moves.

C-MAPSS is the corpus the end-to-end tests publish, so its name, subset and sample have names of
their own here; every other corpus is reached through ``sample`` and ``raw_root``.
"""

from pathlib import Path

from emblema.catalog.application.use_cases.publish_corpus import PublishCorpusCommand
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.entrypoints.cli.known_corpora import KnownCorpora
from scripts.raw_corpora import raw_root as downloaded_root

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = REPO_ROOT / "tests" / "data"
BUDGET = REPO_ROOT / "scripts" / "corpus_budget.toml"

CORPUS = "cmapss"
SUBSET = "FD001"
SAMPLE = SAMPLES / CORPUS
# Not the window a run cuts: short enough that the two sample engines yield several windows each,
# with a stride that leaves a tail beyond the last of them.
SAMPLE_WINDOW = WindowSpec(length=20.0, stride=7.0)


def sample(corpus: str) -> Path:
    """Where the miniature sample of a corpus sits, the root its reader is bound to."""
    return SAMPLES / corpus


def raw_root(corpus: str = CORPUS) -> Path | None:
    """Where the downloaded files sit; ``None`` on a machine that has not fetched them.

    Answered by the same table of markers the reports read, rather than by a copy of it: a test
    that kept its own would stop running against the full corpus the day an archive changed how
    it nests, and say nothing, because a corpus that is not there is a skip.
    """
    return downloaded_root(corpus)


def publish_command(
    *,
    window: WindowSpec = SAMPLE_WINDOW,
    validation_fraction: float = 0.5,
    seed: int = 1,
    corpus: str = CORPUS,
) -> PublishCorpusCommand:
    """The command a run issues: the corpus's own facts, plus how this test wants it cut."""
    known = KnownCorpora.default().named(corpus)
    return PublishCorpusCommand(
        name=known.name,
        source=known.source,
        licence=known.licence,
        window=window,
        validation_fraction=validation_fraction,
        seed=seed,
    )
