from collections.abc import Mapping
from dataclasses import dataclass, field

from torch import Tensor

RECONSTRUCTION = "reconstruction"
"""The name of the hidden tokens' own term in an objective of several."""


@dataclass(frozen=True)
class ObjectiveScore:
    """What one batch cost an objective, before it is divided: three sums over the same tokens.

    The error carries the graph the step is taken on; the trivial predictor's loss is what the
    same tokens would have cost a model that learnt nothing, read the same way, and is detached;
    the count is what divides either into a mean. Summed rather than averaged so that whoever
    accumulates several batches into one gradient adds the sums and divides once, and a batch
    that hid three tokens weighs a hundredth of one that hid three hundred.

    An objective of more than one term names its summands beside the sums, so a log can say what
    each term cost; an objective of one term names none. The named sums are detached: they are
    read, never stepped on. Every such objective names the hidden tokens' own term
    ``reconstruction``, so a run of it can be read on that term as a run without the others is.

    Raises:
        ValueError: If the terms and their trivial costs are named differently, or an objective
            of several terms leaves the hidden tokens' own unnamed.

    Attributes:
        error: The run's loss summed over the hidden tokens that were observed.
        trivial: The trivial predictor's loss summed over the same tokens.
        tokens: How many tokens were scored.
        terms: The summands of ``error`` by name, where there is more than one.
        trivial_terms: The summands of ``trivial`` by the same names.
    """

    error: Tensor
    trivial: Tensor
    tokens: Tensor
    terms: Mapping[str, Tensor] = field(default_factory=dict)
    trivial_terms: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.terms.keys() != self.trivial_terms.keys():
            raise ValueError(
                f"the terms {sorted(self.terms)} and their trivial costs "
                f"{sorted(self.trivial_terms)} must be named alike"
            )
        if self.terms and RECONSTRUCTION not in self.terms:
            raise ValueError(
                f"an objective of several terms names the hidden tokens' own "
                f"{RECONSTRUCTION!r}, got {sorted(self.terms)}"
            )

    @property
    def reconstruction(self) -> tuple[Tensor, Tensor]:
        """The hidden tokens' own error and trivial cost, without any term beside them.

        The whole of either sum under an objective of one term, so a run under a second term is
        scored on what a run without it is scored on.
        """
        if self.terms:
            return self.terms[RECONSTRUCTION], self.trivial_terms[RECONSTRUCTION]
        return self.error.detach(), self.trivial
