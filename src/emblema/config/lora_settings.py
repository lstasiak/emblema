from pydantic import BaseModel, Field


class LoraSettings(BaseModel):
    """The low-rank updates the arm of that name adds beside the backbone's frozen layers.

    Nothing has a default, for the reason the schedule has none: which layers are updated and
    how strongly is a cap on the degrees of freedom the curve is read against. What each field
    means is ``LoraSpec``'s to say.

    The layers are one comma-separated string rather than a list, because a list read from the
    environment is decoded as JSON before anything of ours sees it, and a JSON array reaches a
    process only if every quote in it survives the shell, file or compose block it was written
    in. The entrypoint splits it; ``LoraSpec`` judges what comes out.
    """

    rank: int
    alpha: float
    dropout: float
    targets: str = Field(
        description=(
            "Comma-separated layers, each a dotted run of consecutive segments of a layer's "
            "path in the backbone."
        )
    )
