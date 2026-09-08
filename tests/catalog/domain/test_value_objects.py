"""Invariants of the small value objects: licence, source, content."""

import pytest

from emblema.catalog.domain.corpus_content import CorpusContent
from emblema.catalog.domain.corpus_source import CorpusSource
from emblema.catalog.domain.exceptions import (
    CatalogError,
    InvalidCorpusContentError,
    InvalidCorpusSourceError,
    InvalidLicenceError,
)
from emblema.catalog.domain.licence import Licence
from emblema.shared.kernel.checksums import Checksum


@pytest.mark.parametrize("identifier", ["", "   "])
def test_licence_rejects_blank_identifier(identifier: str) -> None:
    with pytest.raises(InvalidLicenceError):
        Licence(identifier, permits_derivatives=False)


@pytest.mark.parametrize(("name", "uri"), [("", "https://x"), ("NASA", ""), (" ", " ")])
def test_source_rejects_blank_fields(name: str, uri: str) -> None:
    with pytest.raises(InvalidCorpusSourceError):
        CorpusSource(name, uri)


def test_content_requires_at_least_one_record() -> None:
    with pytest.raises(InvalidCorpusContentError, match="positive"):
        CorpusContent(Checksum.of_bytes(b"x"), 0)


def test_invariant_errors_are_both_catalog_and_value_errors() -> None:
    with pytest.raises(CatalogError):
        Licence("", permits_derivatives=False)
    with pytest.raises(ValueError, match="non-blank"):
        Licence("", permits_derivatives=False)
