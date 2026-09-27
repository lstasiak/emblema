"""A task over outcomes, and the answers of every cell kept beside its errors.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "evaluation"
TABLE = "downstream_task"


def upgrade() -> None:
    op.add_column(TABLE, sa.Column("label_outcome", sa.Text(), nullable=True), schema=SCHEMA)
    op.drop_constraint("ck_downstream_task_label_scheme_known", TABLE, schema=SCHEMA)
    op.create_check_constraint(
        "ck_downstream_task_label_scheme_known",
        TABLE,
        "label_scheme IS NULL OR label_scheme IN ('remaining_life', 'forecast', 'outcome')",
        schema=SCHEMA,
    )
    # A task over outcomes is spread over its two outcomes, which has no count to store, so the
    # count of strata now goes with the schemes that read a quantity rather than with any scheme.
    op.drop_constraint("ck_downstream_task_strata_with_labels", TABLE, schema=SCHEMA)
    op.create_check_constraint(
        "ck_downstream_task_strata_with_quantities",
        TABLE,
        "COALESCE(label_scheme IN ('remaining_life', 'forecast'), FALSE) = (strata IS NOT NULL)",
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_downstream_task_outcome_has_name",
        TABLE,
        "(label_scheme = 'outcome') = (label_outcome IS NOT NULL)",
        schema=SCHEMA,
    )
    op.create_table(
        "campaign_window_prediction",
        sa.Column("campaign_id", sa.Uuid(), nullable=False),
        sa.Column("candidate", sa.Text(), nullable=False),
        sa.Column("budget", sa.Text(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("unit", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("ends_at", sa.Float(), nullable=False),
        sa.Column("target", sa.Float(), nullable=False),
        sa.Column("predicted", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(
            ["campaign_id", "candidate", "budget", "seed"],
            [
                f"{SCHEMA}.campaign_cell.campaign_id",
                f"{SCHEMA}.campaign_cell.candidate",
                f"{SCHEMA}.campaign_cell.budget",
                f"{SCHEMA}.campaign_cell.seed",
            ],
            name="fk_campaign_window_prediction_campaign_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "campaign_id",
            "candidate",
            "budget",
            "seed",
            "unit",
            "position",
            name="pk_campaign_window_prediction",
        ),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("campaign_window_prediction", schema=SCHEMA)
    op.drop_constraint("ck_downstream_task_outcome_has_name", TABLE, schema=SCHEMA)
    op.drop_constraint("ck_downstream_task_strata_with_quantities", TABLE, schema=SCHEMA)
    op.create_check_constraint(
        "ck_downstream_task_strata_with_labels",
        TABLE,
        "(label_scheme IS NULL) = (strata IS NULL)",
        schema=SCHEMA,
    )
    op.drop_constraint("ck_downstream_task_label_scheme_known", TABLE, schema=SCHEMA)
    op.create_check_constraint(
        "ck_downstream_task_label_scheme_known",
        TABLE,
        "label_scheme IS NULL OR label_scheme IN ('remaining_life', 'forecast')",
        schema=SCHEMA,
    )
    op.drop_column(TABLE, "label_outcome", schema=SCHEMA)
