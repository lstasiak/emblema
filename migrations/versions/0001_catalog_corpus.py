"""The Catalog's corpus and its versions.

Revision ID: 0001
Revises:
Create Date: 2026-09-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "catalog"


def upgrade() -> None:
    # The local stack creates the schema when its volume is first initialised; a database that
    # was not started that way gets it here, so the migration stands on its own.
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    op.create_table(
        "corpus",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_corpus"),
        sa.UniqueConstraint("name", name="uq_corpus_name"),
        schema=SCHEMA,
    )
    op.create_table(
        "corpus_version",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("corpus_id", sa.Uuid(), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("channel_schema", postgresql.JSONB(), nullable=False),
        sa.Column("sampling_regime", sa.Text(), nullable=False),
        sa.Column("licence_identifier", sa.Text(), nullable=False),
        sa.Column("licence_permits_derivatives", sa.Boolean(), nullable=False),
        sa.Column("licence_url", sa.Text(), nullable=True),
        sa.Column("checksum_algorithm", sa.Text(), nullable=True),
        sa.Column("checksum_digest", sa.Text(), nullable=True),
        sa.Column("unit_count", sa.Integer(), nullable=True),
        sa.Column("observation_count", sa.Integer(), nullable=True),
        sa.Column("frozen_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "frozen_at IS NULL OR checksum_digest IS NOT NULL",
            name="ck_corpus_version_frozen_has_content",
        ),
        sa.ForeignKeyConstraint(
            ["corpus_id"],
            [f"{SCHEMA}.corpus.id"],
            name="fk_corpus_version_corpus_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_corpus_version"),
        sa.UniqueConstraint("corpus_id", "number", name="uq_corpus_version_corpus_id"),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("corpus_version", schema=SCHEMA)
    op.drop_table("corpus", schema=SCHEMA)
