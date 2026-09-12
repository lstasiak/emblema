from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Self

from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.catalog.domain.channel_schema import ChannelSchema
from emblema.catalog.domain.corpus_content import CorpusContent
from emblema.catalog.domain.corpus_description import CorpusDescription
from emblema.catalog.domain.exceptions import (
    CorpusVersionFrozenError,
    CorpusVersionNotFrozenError,
    CorpusVersionNotValidatedError,
    InvalidCorpusVersionError,
)
from emblema.catalog.domain.licence import Licence
from emblema.shared.kernel.sampling import SamplingRegime
from emblema.shared.kernel.timestamps import UtcDateTime


@dataclass(frozen=True)
class CorpusVersion:
    """One snapshot of the data of a corpus, described by its channel schema and sampling regime.

    A version starts as a draft: its content (checksum, record count) is recorded once the data
    has been validated and may be re-recorded while the data is still being prepared. Freezing
    makes the version immutable: any later change is a domain error and requires a new version.

    Invariants: a frozen version always has content; the version number is positive.

    Attributes:
        id: Identity of the version, referenced by other contexts.
        number: Position in the version sequence of the corpus, starting at 1.
        channel_schema: Channels the data carries.
        sampling_regime: How the data is spaced in time.
        licence: Terms the data was obtained under.
        content: Result of validating the data; ``None`` until validated.
        frozen_at: When the version was frozen; ``None`` while it is a draft.
    """

    id: CorpusVersionId
    number: int
    channel_schema: ChannelSchema
    sampling_regime: SamplingRegime
    licence: Licence
    content: CorpusContent | None = None
    frozen_at: UtcDateTime | None = None

    def __post_init__(self) -> None:
        if self.number < 1:
            raise InvalidCorpusVersionError(f"version number must be positive, got {self.number}")
        if self.frozen_at is not None and self.content is None:
            raise InvalidCorpusVersionError(f"frozen version {self.number} must have content")

    @property
    def is_frozen(self) -> bool:
        return self.frozen_at is not None

    def frozen_content(self) -> CorpusContent:
        """Content of the version once frozen: the only content other contexts may rely on.

        Raises:
            CorpusVersionNotFrozenError: If the version is still a draft, with or without content.
        """
        # A frozen version always has content; the second test only narrows the type.
        if self.frozen_at is None or self.content is None:
            raise CorpusVersionNotFrozenError(
                f"version {self.number} is a draft; only frozen versions are published"
            )
        return self.content

    def describes(self, description: CorpusDescription) -> bool:
        """Whether this version carries validated data equal to what ``description`` describes.

        Schema and regime are part of what a version describes: the same bytes read under another
        channel schema or sampling regime are different data. A version without content describes
        nothing yet, so it never matches.
        """
        if self.content is None:
            return False
        return (
            self.content.checksum == description.content.checksum
            and self.channel_schema == description.channel_schema
            and self.sampling_regime == description.sampling_regime
        )

    def describes_same_data_as(self, other: CorpusVersion) -> bool:
        """Whether both versions carry validated data of equal checksum, schema and regime."""
        if other.content is None:
            return False
        return self.describes(
            CorpusDescription(other.channel_schema, other.sampling_regime, other.content)
        )

    def with_content(self, content: CorpusContent) -> Self:
        """Record the outcome of validating the data.

        Raises:
            CorpusVersionFrozenError: If the version is frozen.
        """
        self._require_draft()
        return replace(self, content=content)

    def freeze(self, at: UtcDateTime) -> Self:
        """Make the version immutable.

        Raises:
            CorpusVersionFrozenError: If the version is already frozen.
            CorpusVersionNotValidatedError: If no content has been recorded.
        """
        self._require_draft()
        if self.content is None:
            raise CorpusVersionNotValidatedError(f"version {self.number} has no validated content")
        return replace(self, frozen_at=at)

    def _require_draft(self) -> None:
        if self.is_frozen:
            raise CorpusVersionFrozenError(
                f"version {self.number} is frozen; changes need a new version"
            )
