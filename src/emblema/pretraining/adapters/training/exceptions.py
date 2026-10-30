"""Failures of the training adapters, and what reading foreign bytes looks like on the way in.

A port speaks the domain's exceptions; the one below is what a reader of stored bytes needs and
no caller of the port can see, so it stays here rather than in the domain.
"""

import pickle
import zipfile

from emblema.pretraining.domain.exceptions import PretrainingError

# What comes back from handing torch bytes it did not write, or state that is not the state
# expected. The reasons differ and to a caller they are one thing: this artifact is not ours.
UNREADABLE_BYTES = (
    KeyError,
    IndexError,
    TypeError,
    ValueError,
    RuntimeError,
    EOFError,
    pickle.UnpicklingError,
    zipfile.BadZipFile,
)


class UnreadableTrainedModelError(PretrainingError):
    """Raised where stored bytes are not a trained model this can rebuild."""


class UnanswerablePretextError(PretrainingError):
    """Raised where a stored objective cannot be asked what it fills in.

    It reads no cells, its bytes do not say what it hid, or a window holds a channel it learnt
    no row for.
    """
