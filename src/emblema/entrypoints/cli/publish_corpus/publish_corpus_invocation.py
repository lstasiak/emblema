from dataclasses import dataclass, field
from pathlib import Path

from emblema.catalog.adapters.readers.utsd import UtsdReading
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
        excluded_units: Units cut from the corpus, by the names the corpus gives their files; a
            downstream task's frozen side.
        reading: How a dataset of the time-series collection is read; nothing else reads it.
    """

    command: PublishCorpusCommand
    corpus_root: Path
    workspace: Path
    subsets: tuple[str, ...]
    per_condition: bool = False
    excluded_units: tuple[str, ...] = ()
    reading: UtsdReading = field(default_factory=UtsdReading)
