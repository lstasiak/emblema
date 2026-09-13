import pytest
from pydantic import ValidationError

from emblema.catalog.adapters.synthetic.dials import Dials
from emblema.catalog.adapters.synthetic.layouts import CONTROL_A, CONTROL_PROCESS


@pytest.mark.parametrize(
    "dials", [CONTROL_A, CONTROL_PROCESS], ids=lambda dials: type(dials).__name__
)
def test_a_dial_these_settings_do_not_have_is_refused_rather_than_ignored(dials: Dials) -> None:
    # Ignoring it would hand back the settings untouched, and a preset that was meant to be
    # derived from would go on being used as if the dial had been turned.
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        dials.with_dials(couplng=0.0)
