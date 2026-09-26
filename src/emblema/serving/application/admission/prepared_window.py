from dataclasses import dataclass

from emblema.serving.domain.admitted_window import AdmittedWindow
from emblema.shared.kernel.tokens import TokenWindow


@dataclass(frozen=True, kw_only=True)
class PreparedWindow:
    """One window ready for the runtime: its tokens, what was admitted, and what to warn of.

    Attributes:
        admitted: The readings the model takes, and the channels dropped on the way.
        tokens: The window as the model is fed it.
        warnings: Every way the answer will be less than what was asked, in words.
    """

    admitted: AdmittedWindow
    tokens: TokenWindow
    warnings: tuple[str, ...]
