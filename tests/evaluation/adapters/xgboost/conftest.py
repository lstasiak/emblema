import pytest

from tests.support.openmp import skip_if_torch_shares_the_process


@pytest.fixture(autouse=True)
def _one_openmp_runtime() -> None:
    skip_if_torch_shares_the_process()
