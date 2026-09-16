from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

from emblema.shared.adapters.persistence.naming import NAMING_CONVENTION


class Base(DeclarativeBase):
    """Declarative base of the Catalog's persistence model: its schema and constraint names.

    Every record class of the Catalog derives from it, so every table lands in the ``catalog``
    schema and every constraint is named by one convention, which is what lets a migration be
    compared with the model constraint by constraint.
    """

    metadata = MetaData(schema="catalog", naming_convention=NAMING_CONVENTION)
