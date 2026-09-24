"""The Serving context's projection of what campaigns kept, and the models promoted out of it.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "serving"


def upgrade() -> None:
    # The local stack creates the schema when its volume is first initialised; a database that
    # was not started that way gets it here, so the migration stands on its own.
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    op.create_table(
        "promotable_artifact",
        sa.Column("campaign_ref", sa.Uuid(), nullable=False),
        sa.Column("candidate", sa.Text(), nullable=False),
        sa.Column("task_ref", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("standing", sa.Text(), nullable=False),
        sa.Column("artifact_key", sa.Text(), nullable=False),
        sa.Column("artifact_algorithm", sa.Text(), nullable=False),
        sa.Column("artifact_digest", sa.Text(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "kind IN ('neural', 'classical')", name="ck_promotable_artifact_kind_known"
        ),
        sa.CheckConstraint(
            "standing IN ('control', 'established', 'not_established', 'worse')",
            name="ck_promotable_artifact_standing_known",
        ),
        sa.PrimaryKeyConstraint("campaign_ref", "candidate", name="pk_promotable_artifact"),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_promotable_artifact_artifact_digest",
        "promotable_artifact",
        ["artifact_digest"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_table(
        "campaign_score",
        sa.Column("campaign_ref", sa.Uuid(), nullable=False),
        sa.Column("candidate", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("metric", sa.Text(), nullable=False),
        sa.Column("budget", sa.Integer(), nullable=True),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("repeats", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "budget IS NULL OR budget >= 1", name="ck_campaign_score_budget_counts_windows"
        ),
        sa.CheckConstraint("repeats >= 1", name="ck_campaign_score_repeated"),
        sa.ForeignKeyConstraint(
            ["campaign_ref", "candidate"],
            [
                f"{SCHEMA}.promotable_artifact.campaign_ref",
                f"{SCHEMA}.promotable_artifact.candidate",
            ],
            name="fk_campaign_score_campaign_ref",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("campaign_ref", "candidate", "position", name="pk_campaign_score"),
        schema=SCHEMA,
    )
    op.create_table(
        "served_model",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("campaign_ref", sa.Uuid(), nullable=False),
        sa.Column("task_ref", sa.Uuid(), nullable=False),
        sa.Column("candidate", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("artifact_key", sa.Text(), nullable=False),
        sa.Column("artifact_algorithm", sa.Text(), nullable=False),
        sa.Column("artifact_digest", sa.Text(), nullable=False),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("withdrawn_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("kind IN ('neural', 'classical')", name="ck_served_model_kind_known"),
        sa.CheckConstraint(
            "withdrawn_at IS NULL OR withdrawn_at >= promoted_at",
            name="ck_served_model_withdrawn_after_promoted",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_served_model"),
        schema=SCHEMA,
    )
    op.create_index(
        "uq_served_model_serving_artifact",
        "served_model",
        ["artifact_algorithm", "artifact_digest"],
        unique=True,
        schema=SCHEMA,
        postgresql_where=sa.text("withdrawn_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_served_model_serving_artifact", table_name="served_model", schema=SCHEMA)
    op.drop_table("served_model", schema=SCHEMA)
    op.drop_table("campaign_score", schema=SCHEMA)
    op.drop_index(
        "ix_promotable_artifact_artifact_digest", table_name="promotable_artifact", schema=SCHEMA
    )
    op.drop_table("promotable_artifact", schema=SCHEMA)
