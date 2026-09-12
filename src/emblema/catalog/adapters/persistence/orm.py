from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base of the Catalog's persistence model: its schema and constraint names.

    Every record class of the Catalog derives from it, so every table lands in the ``catalog``
    schema and every constraint is named by one convention, which is what lets a migration be
    compared with the model constraint by constraint.
    """

    metadata = MetaData(
        schema="catalog",
        naming_convention={
            "pk": "pk_%(table_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "ix": "ix_%(table_name)s_%(column_0_name)s",
        },
    )
