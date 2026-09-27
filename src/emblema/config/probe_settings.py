from pydantic import BaseModel, Field


class ProbeSettings(BaseModel):
    """What the probe whose head is solved in closed form chooses its penalty among.

    Nothing has a default, for the reason the schedule has none: the grid a fit chooses from is
    part of what it answers. Its own grid rather than the convolution baseline's, because the
    baseline's is read over ten thousand features and the probe's over the encoder's width.
    What the field means is ``RidgePenalties``' to say.
    """

    ridge_penalties: str = Field(
        description="Comma-separated penalties the ridge fit chooses among, ascending."
    )
