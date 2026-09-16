import pytest

from emblema.pretraining.domain.exceptions import InvalidPretrainingInputError
from tests.support.handoff import pretraining_input


@pytest.mark.parametrize(
    ("field", "value"),
    [("corpus", ""), ("corpus", " cmapss"), ("corpus", "cmapss "), ("vocabulary_size", 0)],
)
def test_an_input_no_run_could_read_is_refused(field: str, value: object) -> None:
    with pytest.raises(InvalidPretrainingInputError, match=field):
        pretraining_input(**{field: value})
