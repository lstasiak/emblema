import pytest

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402

from emblema.pretraining.adapters.objective.objective_score import (  # noqa: E402
    RECONSTRUCTION,
    ObjectiveScore,
)

pytestmark = pytest.mark.ml


def sums(*values: float) -> tuple[Tensor, ...]:
    return tuple(torch.tensor(value) for value in values)


def test_an_objective_of_one_term_is_read_whole() -> None:
    error, trivial, tokens = sums(3.0, 4.0, 5.0)

    read = ObjectiveScore(error=error, trivial=trivial, tokens=tokens).reconstruction

    assert [float(part) for part in read] == [3.0, 4.0]


def test_an_objective_of_several_terms_is_read_on_the_hidden_tokens_own() -> None:
    error, trivial, tokens, own, own_trivial, other, other_trivial = sums(
        3.0, 4.0, 5.0, 1.0, 1.5, 2.0, 2.5
    )

    read = ObjectiveScore(
        error=error,
        trivial=trivial,
        tokens=tokens,
        terms={RECONSTRUCTION: own, "other": other},
        trivial_terms={RECONSTRUCTION: own_trivial, "other": other_trivial},
    ).reconstruction

    assert [float(part) for part in read] == [1.0, 1.5]


def test_an_objective_of_several_terms_that_leaves_its_own_unnamed_is_refused() -> None:
    error, trivial, tokens, one, two = sums(3.0, 4.0, 5.0, 1.0, 2.0)

    with pytest.raises(ValueError, match="reconstruction"):
        ObjectiveScore(
            error=error,
            trivial=trivial,
            tokens=tokens,
            terms={"values": one, "other": two},
            trivial_terms={"values": one, "other": two},
        )


def test_terms_whose_trivial_costs_are_named_otherwise_are_refused() -> None:
    error, trivial, tokens, one, two = sums(3.0, 4.0, 5.0, 1.0, 2.0)

    with pytest.raises(ValueError, match="named alike"):
        ObjectiveScore(
            error=error,
            trivial=trivial,
            tokens=tokens,
            terms={RECONSTRUCTION: one, "other": two},
            trivial_terms={RECONSTRUCTION: one},
        )
