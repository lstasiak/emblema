"""The Evaluation context's tasks and the campaigns run over them.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "evaluation"


def upgrade() -> None:
    # The local stack creates the schema when its volume is first initialised; a database that
    # was not started that way gets it here, so the migration stands on its own.
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    op.create_table(
        "downstream_task",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("corpus", sa.Text(), nullable=False),
        sa.Column("manifest_key", sa.Text(), nullable=False),
        sa.Column("manifest_algorithm", sa.Text(), nullable=False),
        sa.Column("manifest_digest", sa.Text(), nullable=False),
        sa.Column("protocol", sa.Text(), nullable=False),
        sa.Column("test_source", sa.Text(), nullable=False),
        sa.Column("label_scheme", sa.Text(), nullable=True),
        sa.Column("label_ceiling", sa.Float(), nullable=True),
        sa.Column("label_channel", sa.Text(), nullable=True),
        sa.Column("label_horizon", sa.Float(), nullable=True),
        sa.Column("strata", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "protocol IN ('label_budget', 'anomaly_detection')",
            name="ck_downstream_task_protocol_known",
        ),
        sa.CheckConstraint(
            "label_scheme IS NULL OR label_scheme IN ('remaining_life', 'forecast')",
            name="ck_downstream_task_label_scheme_known",
        ),
        sa.CheckConstraint(
            "(protocol = 'label_budget') = (label_scheme IS NOT NULL)",
            name="ck_downstream_task_labels_match_protocol",
        ),
        sa.CheckConstraint(
            "(label_scheme IS NULL) = (strata IS NULL)",
            name="ck_downstream_task_strata_with_labels",
        ),
        sa.CheckConstraint(
            "(label_scheme = 'remaining_life') = (label_ceiling IS NOT NULL)",
            name="ck_downstream_task_remaining_life_has_ceiling",
        ),
        sa.CheckConstraint(
            "(label_scheme = 'forecast') = (label_channel IS NOT NULL)",
            name="ck_downstream_task_forecast_has_channel",
        ),
        sa.CheckConstraint(
            "(label_scheme = 'forecast') = (label_horizon IS NOT NULL)",
            name="ck_downstream_task_forecast_has_horizon",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_downstream_task"),
        schema=SCHEMA,
    )
    op.create_table(
        "task_unit",
        sa.Column("task_id", sa.Uuid(), nullable=False),
        sa.Column("unit", sa.Text(), nullable=False),
        sa.Column("side", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "side IN ('tuning', 'validation', 'test')", name="ck_task_unit_side_known"
        ),
        sa.ForeignKeyConstraint(
            ["task_id"],
            [f"{SCHEMA}.downstream_task.id"],
            name="fk_task_unit_task_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("task_id", "unit", name="pk_task_unit"),
        schema=SCHEMA,
    )
    op.create_table(
        "evaluation_campaign",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("task_ref", sa.Uuid(), nullable=False),
        sa.Column("purpose", sa.Text(), nullable=False),
        sa.Column("tier", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("design", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'finished')", name="ck_evaluation_campaign_status_known"
        ),
        sa.CheckConstraint(
            "(status = 'finished') = (completed_at IS NOT NULL)",
            name="ck_evaluation_campaign_finished_is_dated",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_evaluation_campaign"),
        schema=SCHEMA,
    )
    op.create_table(
        "campaign_cell",
        sa.Column("campaign_id", sa.Uuid(), nullable=False),
        sa.Column("candidate", sa.Text(), nullable=False),
        sa.Column("budget", sa.Text(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("seconds", sa.Float(), nullable=False),
        sa.Column("artifact_key", sa.Text(), nullable=True),
        sa.Column("artifact_algorithm", sa.Text(), nullable=True),
        sa.Column("artifact_digest", sa.Text(), nullable=True),
        sa.CheckConstraint("seconds >= 0", name="ck_campaign_cell_seconds_not_negative"),
        sa.CheckConstraint(
            "num_nonnulls(artifact_key, artifact_algorithm, artifact_digest) IN (0, 3)",
            name="ck_campaign_cell_artifact_is_whole",
        ),
        sa.ForeignKeyConstraint(
            ["campaign_id"],
            [f"{SCHEMA}.evaluation_campaign.id"],
            name="fk_campaign_cell_campaign_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "campaign_id", "candidate", "budget", "seed", name="pk_campaign_cell"
        ),
        schema=SCHEMA,
    )
    op.create_table(
        "campaign_unit_error",
        sa.Column("campaign_id", sa.Uuid(), nullable=False),
        sa.Column("candidate", sa.Text(), nullable=False),
        sa.Column("budget", sa.Text(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("unit", sa.Text(), nullable=False),
        sa.Column("squared_error", sa.Float(), nullable=False),
        sa.Column("windows", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["campaign_id", "candidate", "budget", "seed"],
            [
                f"{SCHEMA}.campaign_cell.campaign_id",
                f"{SCHEMA}.campaign_cell.candidate",
                f"{SCHEMA}.campaign_cell.budget",
                f"{SCHEMA}.campaign_cell.seed",
            ],
            name="fk_campaign_unit_error_campaign_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "campaign_id", "candidate", "budget", "seed", "unit", name="pk_campaign_unit_error"
        ),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("campaign_unit_error", schema=SCHEMA)
    op.drop_table("campaign_cell", schema=SCHEMA)
    op.drop_table("evaluation_campaign", schema=SCHEMA)
    op.drop_table("task_unit", schema=SCHEMA)
    op.drop_table("downstream_task", schema=SCHEMA)
