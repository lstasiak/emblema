from enum import StrEnum


class Retention(StrEnum):
    """How long a stored artifact is kept.

    Attributes:
        DURABLE: Kept until removed on purpose. The class of every artifact the domain
            registers, so a reference recorded in the database keeps resolving.
        TRANSIENT: Intermediate output such as a checkpoint written mid-training. The store
            may expire it by a lifecycle rule once it is older than the configured age.
    """

    DURABLE = "durable"
    TRANSIENT = "transient"
