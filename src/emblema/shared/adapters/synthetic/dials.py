from typing import Self

from pydantic import BaseModel, ConfigDict


class Dials(BaseModel):
    """Settings of the synthetic control: frozen, constraining each other, derived by turning one.

    A model rather than a dataclass, because the fields constrain each other and a value that
    breaks a constraint must not be constructible. Derivation goes through validation for the
    same reason: a layout derived from a valid one is not automatically valid itself, so
    ``with_dials`` revalidates where ``model_copy`` would wave the constraints through. A dial
    the model does not have is refused for the same reason: a misspelt one would otherwise
    return the settings untouched, and a derivation that changed nothing reads as one that
    worked.

    The presets themselves stay where they are stated. This is only the shape they share.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def with_dials(self, **dials: object) -> Self:
        """The same settings with those dials turned and every other one untouched."""
        return type(self).model_validate({**self.model_dump(), **dials})
