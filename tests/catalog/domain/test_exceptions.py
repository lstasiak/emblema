import pytest

from emblema.catalog.domain.exceptions import CatalogError
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.registry.licence import Licence


def test_invariant_errors_are_both_catalog_and_value_errors() -> None:
    with pytest.raises(CatalogError):
        Licence("", permits_derivatives=False)
    with pytest.raises(ValueError, match="non-blank"):
        Licence("", permits_derivatives=False)
    with pytest.raises(ValueError, match="finite"):
        Observation("T2", 0.0, float("nan"))
