from pydantic import BaseModel


class PatchSettings(BaseModel):
    """How the patch model reads a window and how large it is.

    Nothing has a default, for the reason the boosting knobs have none. What each field means is
    ``PatchModelSpec``'s to say, and it is that value object that judges what arrives here.
    """

    patch_length: int
    stride: int
    width: int
    heads: int
    layers: int
    feedforward_width: int
    dropout: float
    grid_resolution: float
