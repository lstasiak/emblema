"""The revision a campaign stands at, so two processes cannot write over each other's cells.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "evaluation"

# The same count the aggregate makes of itself: one for every cell recorded and one more once
# the grid was closed. Stated here rather than defaulted to zero, so a campaign that was already
# under way is claimable at the revision it actually stands at.
STANDING_AT = f"""
    UPDATE {SCHEMA}.evaluation_campaign AS c
    SET version = (
        SELECT count(*) FROM {SCHEMA}.campaign_cell AS x WHERE x.campaign_id = c.id
    ) + CASE WHEN c.completed_at IS NULL THEN 0 ELSE 1 END
"""


def upgrade() -> None:
    op.add_column(
        "evaluation_campaign",
        sa.Column("version", sa.Integer(), nullable=False, server_default="0"),
        schema=SCHEMA,
    )
    op.execute(STANDING_AT)
    # The value belongs to whoever writes the row; a default would let a writer leave it out and
    # claim a revision it never read.
    op.alter_column("evaluation_campaign", "version", server_default=None, schema=SCHEMA)


def downgrade() -> None:
    op.drop_column("evaluation_campaign", "version", schema=SCHEMA)
