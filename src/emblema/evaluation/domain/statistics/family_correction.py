"""How a family of secondary comparisons is corrected, whichever of the known procedures it is.

A closed union rather than a protocol, for the reason the classical methods are one: the
corrections a registration may name are a set this context chooses, and naming them here lets a
codec match on them exhaustively instead of inventing an interface with exactly these two
implementations. Both answer the same question — which comparisons are rejected — and differ in
what they promise about the rejections: Holm bounds the chance of any false one, Benjamini and
Hochberg bound their expected share.
"""

from emblema.evaluation.domain.statistics.benjamini_hochberg_correction import (
    BenjaminiHochbergCorrection,
)
from emblema.evaluation.domain.statistics.holm_correction import HolmCorrection

type FamilyCorrection = HolmCorrection | BenjaminiHochbergCorrection
