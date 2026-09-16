"""The one convention every context's constraints are named by.

Stated once so that a migration of any schema can be compared with its model constraint by
constraint, and so that an adapter recognising a violated constraint by name reads the same
names whichever context it serves.
"""

from typing import Final

NAMING_CONVENTION: Final = {
    "pk": "pk_%(table_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "ix": "ix_%(table_name)s_%(column_0_name)s",
}
