from enum import StrEnum


class ServedModelState(StrEnum):
    """Whether a served model answers requests.

    Attributes:
        SERVING: Promoted and not withdrawn.
        WITHDRAWN: Taken out of service for good. The model is kept, because what was served,
            and from when to when, is the provenance of every answer it gave.
    """

    SERVING = "serving"
    WITHDRAWN = "withdrawn"
