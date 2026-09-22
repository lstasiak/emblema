from dataclasses import dataclass
from pathlib import Path

from emblema.catalog.application.use_cases.publish_corpus import PublishCorpusCommand


@dataclass(frozen=True, kw_only=True)
class PublishCorpusInvocation:
    """One run of the publishing command line: the command and where the process reads and writes.

    Attributes:
        command: What to publish.
        corpus_root: Directory the raw corpus is read from.
        workspace: Directory blocks pass through on their way to the store.
        subsets: Subsets of the corpus to read.
        per_condition: Whether each sensor is read as a channel per operating condition.
    """

    command: PublishCorpusCommand
    corpus_root: Path
    workspace: Path
    subsets: tuple[str, ...]
    per_condition: bool = False
