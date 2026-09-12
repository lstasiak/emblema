import pytest

from emblema.catalog.domain.exceptions import InvalidCorpusSourceError
from emblema.catalog.domain.registry.corpus_source import CorpusSource


@pytest.mark.parametrize(
    ("name", "uri"),
    [("", "https://x"), ("NASA", ""), (" ", " "), (" NASA", "https://x"), ("NASA", "https://x ")],
)
def test_source_rejects_blank_or_padded_fields(name: str, uri: str) -> None:
    with pytest.raises(InvalidCorpusSourceError):
        CorpusSource(name, uri)
