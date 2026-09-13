import pytest

from tests.support.control_corpus import Control, publish_control


@pytest.fixture(scope="session")
def control(tmp_path_factory: pytest.TempPathFactory) -> Control:
    return publish_control(tmp_path_factory.mktemp("control"))
