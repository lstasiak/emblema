from emblema.pretraining.domain.assessment.check import Area, Rule


def test_every_area_is_reached_by_some_rule() -> None:
    assert {rule.area for rule in Rule} == set(Area)
