from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from statistics import fmean, variance

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.exceptions import SelectionNotReadableError


@dataclass(frozen=True)
class OneStandardErrorRule:
    """Chooses the variant closest to its default among those as good as the best within error.

    The rule of one standard error (Breiman, Friedman, Olshen and Stone, 1984; Hastie,
    Tibshirani and Friedman, *The Elements of Statistical Learning*, §7.10). The lowest mean
    error alone would often choose noise: with a few dozen units, the spread between repeats is
    as large as the differences between variants. So every variant whose mean is within one
    standard error of the best one's counts as as good, and among those the one that departs
    least from the candidate's default is chosen — the default wins unless the data say
    clearly otherwise.

    The repeats divide the same units over and over, so their errors are not independent, and
    the plain standard error would understate how uncertain the best mean is. The variance is
    scaled as Nadeau and Bengio (2003) correct it for repeated random holdout: by
    ``1 / J + n_test / n_train`` over ``J`` repeats, rather than by ``1 / J``.

    Ties in closeness are broken by the lower mean and then by name, so a choice never depends
    on the order the variants were listed in.
    """

    def choose(
        self,
        errors: Mapping[CandidateRef, Sequence[float]],
        closeness: Mapping[CandidateRef, tuple[int, float]],
        test_to_train: float,
    ) -> CandidateRef:
        """The variant this rule chooses.

        Args:
            errors: Each variant's error in every repeat, the same repeats for each.
            closeness: How far each variant departs from the default: how many knobs it turns,
                then how far, as the method states it; the default is ``(0, 0.0)``.
            test_to_train: Units scored for every unit learnt from, in each repeat.

        Raises:
            SelectionNotReadableError: If a variant has fewer than two repeats, the variants
                have different repeats, or a variant's closeness is unknown.
        """
        repeats = {len(values) for values in errors.values()}
        if not errors or len(repeats) != 1 or repeats.pop() < 2:
            raise SelectionNotReadableError(
                "every variant needs the same repeats, at least two, for its spread to be read"
            )
        if set(errors) - set(closeness):
            raise SelectionNotReadableError("a variant's departure from its default is unknown")
        means = {variant: fmean(values) for variant, values in errors.items()}
        best = min(means, key=lambda variant: (means[variant], str(variant)))
        spread = errors[best]
        reach = means[best] + sqrt(variance(spread) * (1.0 / len(spread) + test_to_train))
        within = [variant for variant, mean in means.items() if mean <= reach]
        return min(within, key=lambda variant: (closeness[variant], means[variant], str(variant)))
