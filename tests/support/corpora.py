"""The corpora the tests read, and what it takes to publish one.

Two of them: the miniature C-MAPSS sample committed to the repository, which every machine has,
and the full download, which only a machine that fetched it has. Stated once, because a test that
anchors the sample on its own depth in the tree breaks when it moves.
"""

from pathlib import Path

from emblema.catalog.application.use_cases.publish_corpus import PublishCorpusCommand
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.entrypoints.cli.known_corpora import KnownCorpora

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE = REPO_ROOT / "tests" / "data" / "cmapss"
RAW = REPO_ROOT / "data" / "raw" / "cmapss"
BUDGET = REPO_ROOT / "scripts" / "corpus_budget.toml"

CORPUS = "cmapss"
SUBSET = "FD001"
# Not the window a run cuts: short enough that the two sample engines yield several windows each,
# with a stride that leaves a tail beyond the last of them.
SAMPLE_WINDOW = WindowSpec(length=20.0, stride=7.0)


def raw_root() -> Path | None:
    """Where the downloaded C-MAPSS files sit; ``None`` on a machine that has not fetched them."""
    hits = sorted(RAW.rglob("train_FD001.txt")) if RAW.is_dir() else []
    return hits[0].parent if hits else None


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
