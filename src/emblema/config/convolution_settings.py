from pydantic import BaseModel, Field


class ConvolutionSettings(BaseModel):
    """How many convolutions the MiniRocket candidate reads a window with, and how it fits.

    Nothing has a default, for the reason the boosting knobs have none. What each field means is
    ``MiniRocketSpec``'s and ``RidgeSpec``'s to say, and it is those value objects that judge
    what arrives here.
    """

    features: int
    ridge_penalties: str = Field(
        description="Comma-separated penalties the ridge fit chooses among, ascending."
    )
    threads: int
