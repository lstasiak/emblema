import pytest

from emblema.catalog.domain.exceptions import InvalidLicenceError
from emblema.catalog.domain.registry.licence import Licence


@pytest.mark.parametrize("identifier", ["", "   ", " CC-BY-4.0", "CC-BY-4.0 "])
def test_licence_rejects_blank_or_padded_identifier(identifier: str) -> None:
    with pytest.raises(InvalidLicenceError):
        Licence(identifier, permits_derivatives=False)


@pytest.mark.parametrize("url", ["", " ", " https://x", "https://x "])
def test_licence_rejects_blank_or_padded_url(url: str) -> None:
    with pytest.raises(InvalidLicenceError):
        Licence("CC-BY-4.0", permits_derivatives=False, url=url)


def test_licence_without_url_is_valid() -> None:
    assert Licence("CC-BY-4.0", permits_derivatives=False).url is None
