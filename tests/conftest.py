import pytest

from tests.support.openmp import WITHOUT_TORCH, draw_the_boundary


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        WITHOUT_TORCH,
        action="store_true",
        help="refuse torch, so that real gradient-boosted trees can be fitted on macOS",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    draw_the_boundary(without_torch=bool(config.getoption(WITHOUT_TORCH)))
