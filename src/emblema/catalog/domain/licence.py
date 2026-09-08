from dataclasses import dataclass

from emblema.catalog.domain.exceptions import InvalidLicenceError


@dataclass(frozen=True)
class Licence:
    """Terms under which a corpus version may be used.

    Publishing results and keeping a small sample in a public repository are preconditions of
    registering a corpus at all, so they are not modelled. Redistribution of derivative works is
    the one permission that varies between admitted corpora and decides whether a processed
    corpus may be distributed publicly.

    Attributes:
        identifier: Licence name or SPDX identifier, non-blank without surrounding whitespace.
        permits_derivatives: Whether derivative works may be redistributed.
        url: Where the licence text lives, when known; non-blank without surrounding whitespace.
    """

    identifier: str
    permits_derivatives: bool
    url: str | None = None

    def __post_init__(self) -> None:
        if not self.identifier or self.identifier != self.identifier.strip():
            raise InvalidLicenceError(
                "licence identifier must be non-blank without surrounding whitespace"
            )
        if self.url is not None and (not self.url or self.url != self.url.strip()):
            raise InvalidLicenceError(
                "licence url must be non-blank without surrounding whitespace"
            )
