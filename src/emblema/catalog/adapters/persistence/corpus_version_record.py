from datetime import UTC, datetime
from typing import Self
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from emblema.catalog.adapters.persistence.orm import Base
from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.catalog.domain.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.corpus_content import CorpusContent
from emblema.catalog.domain.corpus_version import CorpusVersion
from emblema.catalog.domain.identifiers import CorpusId
from emblema.catalog.domain.licence import Licence
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm
from emblema.shared.kernel.sampling import SamplingRegime
from emblema.shared.kernel.timestamps import UtcDateTime


class CorpusVersionRecord(Base):
    """Row of ``catalog.corpus_version``: one version of a corpus, flattened to columns.

    The channel schema is JSONB, a set of small records whose shape changes more often than it is
    queried. The checksum is two columns because other contexts pin a version by it. Content
    columns are null for a draft; the check constraint repeats what the aggregate enforces.
    """

    __tablename__ = "corpus_version"
    __table_args__ = (
        UniqueConstraint("corpus_id", "number"),
        CheckConstraint(
            "frozen_at IS NULL OR checksum_digest IS NOT NULL", name="frozen_has_content"
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    corpus_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("catalog.corpus.id", ondelete="CASCADE")
    )
    number: Mapped[int] = mapped_column(Integer)
    channel_schema: Mapped[list[dict[str, object]]] = mapped_column(JSONB)
    sampling_regime: Mapped[str] = mapped_column(Text)
    licence_identifier: Mapped[str] = mapped_column(Text)
    licence_permits_derivatives: Mapped[bool] = mapped_column(Boolean)
    licence_url: Mapped[str | None] = mapped_column(Text)
    checksum_algorithm: Mapped[str | None] = mapped_column(Text)
    checksum_digest: Mapped[str | None] = mapped_column(Text)
    unit_count: Mapped[int | None] = mapped_column(Integer)
    observation_count: Mapped[int | None] = mapped_column(Integer)
    frozen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    @classmethod
    def from_version(cls, corpus_id: CorpusId, version: CorpusVersion) -> Self:
        content = version.content
        return cls(
            id=version.id.value,
            corpus_id=corpus_id.value,
            number=version.number,
            channel_schema=[
                {"name": channel.name, "unit": channel.unit, "timeless": channel.timeless}
                for channel in sorted(version.channel_schema, key=lambda channel: channel.name)
            ],
            sampling_regime=str(version.sampling_regime),
            licence_identifier=version.licence.identifier,
            licence_permits_derivatives=version.licence.permits_derivatives,
            licence_url=version.licence.url,
            checksum_algorithm=None if content is None else str(content.checksum.algorithm),
            checksum_digest=None if content is None else content.checksum.digest,
            unit_count=None if content is None else content.unit_count,
            observation_count=None if content is None else content.observation_count,
            frozen_at=None if version.frozen_at is None else version.frozen_at.value,
        )

    def to_version(self) -> CorpusVersion:
        return CorpusVersion(
            id=CorpusVersionId(self.id),
            number=self.number,
            channel_schema=ChannelSchema(
                frozenset(
                    Channel(
                        str(channel["name"]),
                        self._optional_text(channel["unit"]),
                        bool(channel["timeless"]),
                    )
                    for channel in self.channel_schema
                )
            ),
            sampling_regime=SamplingRegime(self.sampling_regime),
            licence=Licence(
                self.licence_identifier, self.licence_permits_derivatives, self.licence_url
            ),
            content=self._content(),
            # The database answers in the session's time zone; the value object requires offset
            # zero, so the boundary that produced the value normalises it.
            frozen_at=None
            if self.frozen_at is None
            else UtcDateTime(self.frozen_at.astimezone(UTC)),
        )

    def _content(self) -> CorpusContent | None:
        if self.checksum_digest is None:
            return None
        return CorpusContent(
            Checksum(HashAlgorithm(str(self.checksum_algorithm)), self.checksum_digest),
            int(self.unit_count or 0),
            int(self.observation_count or 0),
        )

    @staticmethod
    def _optional_text(value: object) -> str | None:
        return None if value is None else str(value)
