from typing import Self
from uuid import UUID

from sqlalchemy import ForeignKey, Integer, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.pretraining.adapters.persistence.orm import Base
from emblema.pretraining.domain.backbone.pretraining_input import PretrainingInput
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm


class PretrainingInputRecord(Base):
    """Row of ``pretraining.pretraining_input``: the published corpus a backbone was trained on.

    The corpus version is a reference into another context's schema, so it is a plain column
    with its checksum beside it rather than a foreign key: what Pretraining knows of the Catalog
    is what the published manifest said, and that is what the row keeps.
    """

    __tablename__ = "pretraining_input"

    backbone_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("pretraining.backbone.id", ondelete="CASCADE"), primary_key=True
    )
    corpus: Mapped[str] = mapped_column(Text)
    corpus_version: Mapped[UUID] = mapped_column(Uuid)
    corpus_checksum_algorithm: Mapped[str] = mapped_column(Text)
    corpus_checksum_digest: Mapped[str] = mapped_column(Text)
    manifest_key: Mapped[str] = mapped_column(Text)
    manifest_algorithm: Mapped[str] = mapped_column(Text)
    manifest_digest: Mapped[str] = mapped_column(Text)
    block_algorithm: Mapped[str] = mapped_column(Text)
    block_digest: Mapped[str] = mapped_column(Text)
    vocabulary_size: Mapped[int] = mapped_column(Integer)

    @classmethod
    def from_input(cls, backbone_id: BackboneId, read: PretrainingInput) -> Self:
        return cls(
            backbone_id=backbone_id.value,
            corpus=read.corpus,
            corpus_version=read.corpus_version.value,
            corpus_checksum_algorithm=str(read.corpus_checksum.algorithm),
            corpus_checksum_digest=read.corpus_checksum.digest,
            manifest_key=read.manifest.key,
            manifest_algorithm=str(read.manifest.checksum.algorithm),
            manifest_digest=read.manifest.checksum.digest,
            block_algorithm=str(read.block_checksum.algorithm),
            block_digest=read.block_checksum.digest,
            vocabulary_size=read.vocabulary_size,
        )

    def to_input(self) -> PretrainingInput:
        return PretrainingInput(
            corpus=self.corpus,
            corpus_version=CorpusVersionId(self.corpus_version),
            corpus_checksum=Checksum(
                HashAlgorithm(self.corpus_checksum_algorithm), self.corpus_checksum_digest
            ),
            manifest=ArtifactRef(
                self.manifest_key,
                Checksum(HashAlgorithm(self.manifest_algorithm), self.manifest_digest),
            ),
            block_checksum=Checksum(HashAlgorithm(self.block_algorithm), self.block_digest),
            vocabulary_size=self.vocabulary_size,
        )
