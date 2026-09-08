from dataclasses import dataclass, replace
from itertools import combinations
from typing import Self

from emblema.catalog.domain.channel_schema import ChannelSchema
from emblema.catalog.domain.corpus_content import CorpusContent
from emblema.catalog.domain.corpus_source import CorpusSource
from emblema.catalog.domain.corpus_version import CorpusVersion
from emblema.catalog.domain.exceptions import (
    CorpusVersionNotFoundError,
    DuplicateCorpusVersionError,
    InvalidCorpusError,
)
from emblema.catalog.domain.identifiers import CorpusId, CorpusVersionId
from emblema.catalog.domain.licence import Licence
from emblema.shared.kernel.sampling import SamplingRegime
from emblema.shared.kernel.timestamps import UtcDateTime


@dataclass(frozen=True)
class Corpus:
    """A named source of measurement data and the sequence of its versions.

    The corpus is the consistency boundary for its versions: numbering, lookup and the rule that
    no two frozen versions describe the same data live here. Every operation returns a new
    corpus; the receiver is never modified.

    Invariants: version numbers are exactly 1..n in order; version identifiers are unique; two
    frozen versions never share checksum, channel schema and sampling regime.

    Attributes:
        id: Identity of the corpus.
        name: Human-readable name, non-blank.
        source: Where the data comes from.
        versions: All versions, drafts and frozen, in version order.
    """

    id: CorpusId
    name: str
    source: CorpusSource
    versions: tuple[CorpusVersion, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise InvalidCorpusError("corpus name must be non-blank")
        numbers = [version.number for version in self.versions]
        if numbers != list(range(1, len(numbers) + 1)):
            raise InvalidCorpusError(f"version numbers must be 1..n in order, got {numbers}")
        if len({version.id for version in self.versions}) != len(self.versions):
            raise InvalidCorpusError("version identifiers must be unique")
        for first, second in combinations(self.frozen_versions, 2):
            if first.describes_same_data_as(second):
                raise InvalidCorpusError("frozen versions must describe distinct data")

    @property
    def frozen_versions(self) -> tuple[CorpusVersion, ...]:
        return tuple(version for version in self.versions if version.is_frozen)

    def get_version(self, version_id: CorpusVersionId) -> CorpusVersion:
        """Look up a version by identifier.

        Raises:
            CorpusVersionNotFoundError: If no version of this corpus has that identifier.
        """
        for version in self.versions:
            if version.id == version_id:
                return version
        raise CorpusVersionNotFoundError(f"corpus {self.id} has no version {version_id}")

    def add_version(
        self,
        version_id: CorpusVersionId,
        channel_schema: ChannelSchema,
        sampling_regime: SamplingRegime,
        licence: Licence,
    ) -> Self:
        """Open a new draft version, numbered after the last one."""
        draft = CorpusVersion(
            id=version_id,
            number=len(self.versions) + 1,
            channel_schema=channel_schema,
            sampling_regime=sampling_regime,
            licence=licence,
        )
        return replace(self, versions=(*self.versions, draft))

    def record_content(self, version_id: CorpusVersionId, content: CorpusContent) -> Self:
        """Record validated content on a draft version.

        Raises:
            CorpusVersionNotFoundError: If the version does not belong to this corpus.
            CorpusVersionFrozenError: If the version is frozen.
        """
        return self._replace_version(self.get_version(version_id).with_content(content))

    def freeze_version(self, version_id: CorpusVersionId, at: UtcDateTime) -> Self:
        """Freeze a validated draft, provided no frozen version already describes the same data.

        Raises:
            CorpusVersionNotFoundError: If the version does not belong to this corpus.
            CorpusVersionFrozenError: If the version is already frozen.
            CorpusVersionNotValidatedError: If the version has no content.
            DuplicateCorpusVersionError: If a frozen version has the same checksum, channel schema
                and sampling regime.
        """
        frozen = self.get_version(version_id).freeze(at)
        for other in self.frozen_versions:
            if other.describes_same_data_as(frozen):
                raise DuplicateCorpusVersionError(
                    f"version {frozen.number} describes the same data as frozen version "
                    f"{other.number}"
                )
        return self._replace_version(frozen)

    def _replace_version(self, version: CorpusVersion) -> Self:
        versions = tuple(version if v.id == version.id else v for v in self.versions)
        return replace(self, versions=versions)
