from dataclasses import dataclass

from emblema.catalog.domain.exceptions import InvalidCorpusSourceError


@dataclass(frozen=True)
class CorpusSource:
    """Canonical origin of a corpus: the institution or author publishing it and where.

    Attributes:
        name: Publisher of the data, non-blank without surrounding whitespace.
        uri: Canonical location or citation of the data, non-blank without surrounding whitespace.
    """

    name: str
    uri: str

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise InvalidCorpusSourceError(
                "source name must be non-blank without surrounding whitespace"
            )
        if not self.uri or self.uri != self.uri.strip():
            raise InvalidCorpusSourceError(
                "source uri must be non-blank without surrounding whitespace"
            )
