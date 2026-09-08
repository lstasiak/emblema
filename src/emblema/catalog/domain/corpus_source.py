from dataclasses import dataclass

from emblema.catalog.domain.exceptions import InvalidCorpusSourceError


@dataclass(frozen=True)
class CorpusSource:
    """Canonical origin of a corpus: the institution or author publishing it and where.

    Attributes:
        name: Publisher of the data, non-blank.
        uri: Canonical location or citation of the data, non-blank.
    """

    name: str
    uri: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise InvalidCorpusSourceError("source name must be non-blank")
        if not self.uri.strip():
            raise InvalidCorpusSourceError("source uri must be non-blank")
