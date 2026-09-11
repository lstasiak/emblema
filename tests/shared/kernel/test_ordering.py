from emblema.shared.kernel.ordering import seeded_rank


def test_the_same_item_under_the_same_seed_ranks_the_same() -> None:
    assert seeded_rank(7, "engine-03") == seeded_rank(7, "engine-03")


def test_another_seed_ranks_the_item_elsewhere() -> None:
    assert seeded_rank(7, "engine-03") != seeded_rank(8, "engine-03")


def test_the_parts_are_read_in_the_order_they_are_given() -> None:
    assert seeded_rank(1, "a", "b") != seeded_rank(1, "b", "a")


def test_a_part_that_looks_like_two_is_not_two() -> None:
    # The framing carries each part's length, so a unit named "3:17" cannot be mistaken for epoch 3
    # at position 17. Joining the parts with a separator made these the same item.
    assert seeded_rank(1, "3:17") != seeded_rank(1, 3, 17)
    assert seeded_rank(1, "a:b") != seeded_rank(1, "a", "b")


def test_an_item_is_ranked_by_how_it_spells_itself() -> None:
    assert seeded_rank(1, 17) == seeded_rank(1, "17")


def test_the_rank_is_the_one_the_rule_produces() -> None:
    # Pinned: every recorded split and every replayed epoch is this function's output, so a change
    # to the framing has to be a deliberate change here rather than a silent reshuffle elsewhere.
    assert seeded_rank(1, 0, 5).hex() == "d67114b8ae401a0b49259b6576906fc5"
    assert seeded_rank(7, "engine-03").hex() == "99a6f0116eb750dc7fd553a0984762e4"
