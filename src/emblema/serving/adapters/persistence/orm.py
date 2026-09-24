from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

from emblema.shared.adapters.persistence.naming import NAMING_CONVENTION


class Base(DeclarativeBase):
    """Declarative base of the Serving context's persistence model: its schema and names.

    Every record class of this context derives from it, so every table lands in the ``serving``
    schema; the constraint names follow the convention every context shares.
    """

    metadata = MetaData(schema="serving", naming_convention=NAMING_CONVENTION)
