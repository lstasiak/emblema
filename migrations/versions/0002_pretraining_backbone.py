"""Pretraining's backbone and the corpus it was trained on.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "pretraining"


def upgrade() -> None:
    # The local stack creates the schema when its volume is first initialised; a database that
    # was not started that way gets it here, so the migration stands on its own.
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    op.create_table(
        "backbone",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("experiment", sa.Text(), nullable=False),
        sa.Column("run", sa.Text(), nullable=False),
        sa.Column("tier", sa.Text(), nullable=False),
        sa.Column("configuration", postgresql.JSONB(), nullable=False),
        sa.Column("parameter_count", sa.Integer(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("git_commit", sa.Text(), nullable=False),
        sa.Column("signature", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("result_key", sa.Text(), nullable=True),
        sa.Column("result_algorithm", sa.Text(), nullable=True),
        sa.Column("result_digest", sa.Text(), nullable=True),
        sa.Column("artifact_key", sa.Text(), nullable=True),
        sa.Column("artifact_algorithm", sa.Text(), nullable=True),
        sa.Column("artifact_digest", sa.Text(), nullable=True),
        sa.Column("ordered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('ordered', 'ready')", name="ck_backbone_status_known"),
        sa.CheckConstraint(
            "(status = 'ready') = (artifact_digest IS NOT NULL)",
            name="ck_backbone_ready_has_artifact",
        ),
        sa.CheckConstraint(
            "(artifact_digest IS NULL) = (delivered_at IS NULL)",
            name="ck_backbone_delivery_dated",
        ),
        sa.CheckConstraint(
            "(artifact_digest IS NULL) = (result_digest IS NULL)",
            name="ck_backbone_delivery_has_result",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_backbone"),
        schema=SCHEMA,
    )
    op.create_table(
        "pretraining_input",
        sa.Column("backbone_id", sa.Uuid(), nullable=False),
        sa.Column("corpus", sa.Text(), nullable=False),
        sa.Column("corpus_version", sa.Uuid(), nullable=False),
        sa.Column("corpus_checksum_algorithm", sa.Text(), nullable=False),
        sa.Column("corpus_checksum_digest", sa.Text(), nullable=False),
        sa.Column("manifest_key", sa.Text(), nullable=False),
        sa.Column("manifest_algorithm", sa.Text(), nullable=False),
        sa.Column("manifest_digest", sa.Text(), nullable=False),
        sa.Column("block_algorithm", sa.Text(), nullable=False),
        sa.Column("block_digest", sa.Text(), nullable=False),
        sa.Column("vocabulary_size", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["backbone_id"],
            [f"{SCHEMA}.backbone.id"],
            name="fk_pretraining_input_backbone_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("backbone_id", name="pk_pretraining_input"),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("pretraining_input", schema=SCHEMA)
    op.drop_table("backbone", schema=SCHEMA)
