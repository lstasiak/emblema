import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.evaluation.domain.exceptions import InvalidWindowRankingError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.scoring.window_ranking import WindowRanking
from tests.evaluation.support import prediction


def ranked(*rows: tuple[str, float, float]) -> WindowRanking:
    return WindowRanking.of(
        prediction(unit, 0, target, predicted) for unit, target, predicted in rows
    )


def by_pairs(rows: list[tuple[float, float, int]]) -> float:
    """The area by its definition: every positive against every negative, weights multiplying."""
    won = total = 0.0
    for positive, answer, weight in rows:
        for negative, other, other_weight in rows:
            if positive == 1.0 and negative == 0.0:
                pairs = weight * other_weight
                total += pairs
                won += pairs * (1.0 if answer > other else 0.5 if answer == other else 0.0)
    return won / total


def test_answers_that_put_every_positive_above_every_negative_rank_perfectly() -> None:
    assert ranked(("a", 0.0, 0.1), ("b", 0.0, 0.2), ("c", 1.0, 0.7), ("d", 1.0, 0.9)).auroc == 1.0


def test_answers_that_put_every_positive_below_every_negative_rank_backwards() -> None:
    assert ranked(("a", 1.0, 0.1), ("b", 0.0, 0.8)).auroc == 0.0


def test_a_constant_answer_ranks_no_pair_and_scores_one_half() -> None:
    assert ranked(("a", 0.0, 0.3), ("b", 1.0, 0.3), ("c", 0.0, 0.3)).auroc == 0.5


def test_a_tie_between_a_positive_and_a_negative_counts_half() -> None:
    # Three positives and four negatives: ten pairs won, one tied and one lost of twelve.
    rows = (
        ("n1", 0.0, 0.1),
        ("n2", 0.0, 0.2),
        ("n3", 0.0, 0.3),
        ("p1", 1.0, 0.4),
        ("n4", 0.0, 0.6),
        ("p2", 1.0, 0.6),
        ("p3", 1.0, 0.9),
    )

    assert ranked(*rows).auroc == pytest.approx(10.5 / 12.0)


def test_units_are_named_in_their_own_order_whatever_order_the_answers_came_in() -> None:
    ranking = ranked(("b", 1.0, 0.9), ("a", 0.0, 0.1))

    assert ranking.units == (UnitKey("a"), UnitKey("b"))


def test_a_ranking_of_one_outcome_is_refused() -> None:
    with pytest.raises(InvalidWindowRankingError, match="both outcomes"):
        ranked(("a", 0.0, 0.1), ("b", 0.0, 0.9))


def test_a_target_that_is_not_an_outcome_is_refused() -> None:
    with pytest.raises(InvalidWindowRankingError, match="not an outcome"):
        ranked(("a", 0.0, 0.1), ("b", 0.5, 0.9))


def test_weights_that_leave_one_outcome_are_refused() -> None:
    ranking = ranked(("a", 0.0, 0.1), ("b", 1.0, 0.9))

    with pytest.raises(InvalidWindowRankingError, match="one outcome only"):
        ranking.auroc_weighted([1, 0])


rows_strategy = st.lists(
    st.tuples(
        st.sampled_from([0.0, 1.0]),
        st.sampled_from([0.0, 0.25, 0.5, 0.75, 1.0]),
        st.integers(min_value=1, max_value=3),
    ),
    min_size=2,
    max_size=12,
).filter(lambda rows: {row[0] for row in rows} == {0.0, 1.0})


@given(rows_strategy)
def test_the_sweep_agrees_with_every_pair_counted_one_by_one(
    rows: list[tuple[float, float, int]],
) -> None:
    predictions = [
        WindowPrediction(
            window=prediction(f"u{index:02d}", 0, 0.0, 0.0).window, target=t, predicted=p
        )
        for index, (t, p, _) in enumerate(rows)
    ]
    ranking = WindowRanking.of(predictions)

    assert ranking.auroc_weighted([weight for _, _, weight in rows]) == pytest.approx(
        by_pairs(rows)
    )


@given(rows_strategy)
def test_a_monotone_transform_of_the_answers_leaves_the_area_where_it_was(
    rows: list[tuple[float, float, int]],
) -> None:
    def ranking_of(transform: float) -> float:
        return WindowRanking.of(
            prediction(f"u{index:02d}", 0, target, 3.0 * answer + transform)
            for index, (target, answer, _) in enumerate(rows)
        ).auroc

    assert ranking_of(0.0) == ranking_of(-2.0)
