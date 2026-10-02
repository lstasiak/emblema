from math import isqrt

from torch import Tensor, nn

from emblema.shared.kernel.tokens import N_FEATURES


class NonlinearValueEmbedding(nn.Module):
    """A token's features through a narrow hidden layer and a tanh before they reach the width.

    The standard encoder maps a token's value and gap to its width by one linear map shared by
    every channel, so a reading's state moves along one direction as its value grows. Here the
    features pass a hidden layer of the width's square root, bent by a tanh, and only then
    spread to the width: a value can land on different directions at different magnitudes, as a
    continuous value embedding does in networks published on irregular clinical series. Those
    embed the value alone; this one takes the token's features as the linear map does, so it
    replaces that map and nothing else.

    Attributes:
        hidden_width: Size of the hidden layer, the integer square root of the width.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.hidden_width = max(isqrt(width), 1)
        self.layers = nn.Sequential(
            nn.Linear(N_FEATURES, self.hidden_width),
            nn.Tanh(),
            nn.Linear(self.hidden_width, width),
        )

    def forward(self, features: Tensor) -> Tensor:
        embedded: Tensor = self.layers(features)
        return embedded
