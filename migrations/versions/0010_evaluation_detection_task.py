"""A detection task reads the events its windows overlap and spends no labels.

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "evaluation"
TABLE = "downstream_task"
SCHEMES = "'remaining_life', 'forecast', 'outcome'"


def upgrade() -> None:
    # Every task stored so far spends labels and carries a scheme, so nothing is null here.
    op.drop_constraint("ck_downstream_task_label_scheme_known", TABLE, schema=SCHEMA)
    op.drop_constraint("ck_downstream_task_labels_match_protocol", TABLE, schema=SCHEMA)
    op.alter_column(TABLE, "label_scheme", existing_type=sa.Text(), nullable=False, schema=SCHEMA)
    op.create_check_constraint(
        "ck_downstream_task_label_scheme_known",
        TABLE,
        f"label_scheme IN ({SCHEMES}, 'anomaly')",
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_downstream_task_labels_match_protocol",
        TABLE,
        "(protocol = 'anomaly_detection') = (label_scheme = 'anomaly')",
        schema=SCHEMA,
    )


def downgrade() -> None:
    # A detection task reads events; the schema before this one has no scheme for them, and
    # dropping the scheme would leave the task claiming a budget axis it does not have.
    detecting = (
        op.get_bind()
        .execute(sa.text(f"SELECT count(*) FROM {SCHEMA}.{TABLE} WHERE label_scheme = 'anomaly'"))
        .scalar_one()
    )
    if detecting:
        raise RuntimeError(
            f"{detecting} tasks in {SCHEMA}.{TABLE} read the events their windows overlap: the "
            "schema before this one has no scheme for them, so downgrading would change what "
            "they ask"
        )
    op.drop_constraint("ck_downstream_task_labels_match_protocol", TABLE, schema=SCHEMA)
    op.drop_constraint("ck_downstream_task_label_scheme_known", TABLE, schema=SCHEMA)
    op.alter_column(TABLE, "label_scheme", existing_type=sa.Text(), nullable=True, schema=SCHEMA)
    op.create_check_constraint(
        "ck_downstream_task_label_scheme_known",
        TABLE,
        f"label_scheme IS NULL OR label_scheme IN ({SCHEMES})",
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_downstream_task_labels_match_protocol",
        TABLE,
        "(protocol = 'label_budget') = (label_scheme IS NOT NULL)",
        schema=SCHEMA,
    )
