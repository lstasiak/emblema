from typing import Any

from emblema.evaluation.domain.exceptions import InvalidComparisonRulesError
from emblema.evaluation.domain.statistics.benjamini_hochberg_correction import (
    BenjaminiHochbergCorrection,
)
from emblema.evaluation.domain.statistics.family_correction import FamilyCorrection
from emblema.evaluation.domain.statistics.holm_correction import HolmCorrection


class FamilyCorrectionDocument:
    """Reads the correction a campaign's rules name to two fields of a document and back.

    Written by the campaign file a comparison is declared in and by the stored design, which
    must agree about what a name means: a campaign declared under one correction and stored
    under another would be read by rules nobody registered. One codec for both keeps that a
    fact rather than a promise. A document written before a correction could be named holds
    Holm, which is what every campaign was read under until then.
    """

    DEFAULT = HolmCorrection.NAME

    def encode(self, correction: FamilyCorrection) -> dict[str, Any]:
        return {"correction": correction.NAME, "alpha": correction.alpha}

    def decode(self, name: str, alpha: float) -> FamilyCorrection:
        """The correction called ``name`` at level ``alpha``.

        Raises:
            InvalidComparisonRulesError: If no correction is called that.
            InvalidFamilyCorrectionError: If the level is not one.
        """
        if name == HolmCorrection.NAME:
            return HolmCorrection(alpha=alpha)
        if name == BenjaminiHochbergCorrection.NAME:
            return BenjaminiHochbergCorrection(alpha=alpha)
        raise InvalidComparisonRulesError(
            f"no family correction is called {name!r}; the rules name "
            f"{HolmCorrection.NAME!r} or {BenjaminiHochbergCorrection.NAME!r}"
        )
