"""How a classical candidate is fitted, whichever of the known methods it is.

A closed union rather than a protocol, for the reason the feature readings are one: the methods
are a set this context chooses, and naming them here lets a runtime match on them exhaustively
instead of inventing an interface that has exactly these implementations and no others. A
method added is a member added here, and every match that forgot it fails the type check.
"""

from emblema.evaluation.domain.classical.boosted_trees import BoostedTrees
from emblema.evaluation.domain.classical.random_convolutions import RandomConvolutions

type ClassicalMethod = BoostedTrees | RandomConvolutions
